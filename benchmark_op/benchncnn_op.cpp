// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
// initialize vulkan runtime before main()
GlobalGpuInstance g_global_gpu_instance;
#endif // NCNN_VULKAN

namespace ncnn {

// always return empty weights
class ModelBinFromEmpty : public ModelBin
{
public:
    // virtual Mat load(int w, int /*type*/) const { return Mat(w); }
    virtual Mat load(int w, int /*type*/) const { Mat weight=Mat(w); weight.fill(0.01f); return weight;}
};

class BenchNet : public Net
{
public:
    int load_model()
    {
        // load file
        int ret = 0;

        ModelBinFromEmpty mb;
        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];

            int lret = layer->load_model(mb);
            if (lret != 0)
            {
                fprintf(stderr, "layer load_model %d failed\n", (int)i);
                ret = -1;
                break;
            }

            int cret = layer->create_pipeline(opt);
            if (cret != 0)
            {
                fprintf(stderr, "layer create_pipeline %d failed\n", (int)i);
                ret = -1;
                break;
            }
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            upload_model();

            create_pipeline();
        }
#endif // NCNN_VULKAN

        fuse_network();

        return ret;
    }
};

} // namespace ncnn

static int g_warmup_loop_count = 3;
static int g_loop_count = 4;

static ncnn::Option g_default_option;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& in)
{
    ncnn::BenchNet net;

    net.opt = g_default_option;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    net.load_model();

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
//     sleep(10);
#endif

    ncnn::Mat out;

    float *ptr = static_cast<float*>(in.data);

    for(unsigned int i = 0; i < in.h * in.w * in.c; i++)
    {
        ptr[i] = 0.01*(i % 10);
    }

    // fprintf(stderr, "first = %7.2f  last = %7.2f\n", ptr[0], ptr[(in.h * in.w * in.c) - 1]);

    // warm up
    for (int i=0; i<g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ex.extract("output", out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("data", in);
            ex.extract("output", out);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }

    bool use_vulkan_compute = gpu_device != -1;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobBufferAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingBufferAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    // default option
    g_default_option.lightmode = true;
    g_default_option.num_threads = num_threads;
    g_default_option.blob_allocator = &g_blob_pool_allocator;
    g_default_option.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    g_default_option.blob_vkallocator = g_blob_vkallocator;
    g_default_option.workspace_vkallocator = g_blob_vkallocator;
    g_default_option.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    g_default_option.use_winograd_convolution = true;
    g_default_option.use_sgemm_convolution = true;
    g_default_option.use_int8_inference = true;
    g_default_option.use_vulkan_compute = use_vulkan_compute;
    g_default_option.use_fp16_packed = true;
    g_default_option.use_fp16_storage = true;
    g_default_option.use_fp16_arithmetic = true;
    g_default_option.use_int8_storage = true;
    g_default_option.use_int8_arithmetic = true;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);

    // convert from tengine-fp32
    // fprintf(stderr, "======models(convert from tengine)-fp32======\n");
    // benchmark("Caffe_MobileNetSSD", ncnn::Mat(300, 300, 3));
    // benchmark("Caffe_YuFaceDetectNet", ncnn::Mat(320, 240, 3));
    // benchmark("Caffe_alexnet", ncnn::Mat(227, 227, 3));
    // benchmark("Caffe_googlenet", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_inception_v3", ncnn::Mat(395, 395, 3));
    // benchmark("Caffe_mnasnet", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_mobilenet_V1.0", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_mobilenetv2_1.0", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_mobilenetv2_yolov3", ncnn::Mat(320, 320, 3));
    // benchmark("Caffe_resnet50", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_shufflenet_1xg3", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_shufflenet_v2", ncnn::Mat(224, 224, 3));
    // benchmark("Caffe_squeezenet_V1_1", ncnn::Mat(227, 227, 3));
    // benchmark("Caffe_squeezenet_ssd", ncnn::Mat(300, 300, 3));
    // benchmark("Caffe_vgg16", ncnn::Mat(224, 224, 3));
    // benchmark("Onnx_squeezenet", ncnn::Mat(227, 227, 3));
    // benchmark("Mxnet_alexnet", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_inception_v3", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_mobilenet_V1.0", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_mobilenet_v2", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_mobilenet_v2_0.25", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_resnet18_v2", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_resnet50", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_squeezenet_V1_1", ncnn::Mat(224, 224, 3));
    // benchmark("Mxnet_vgg16", ncnn::Mat(224, 224, 3));
    

    // convert from tengine-int8
    // fprintf(stderr, "======models(convert from tengine)-int8======\n");
    // benchmark("caffe_mssd-int8", ncnn::Mat(300, 300, 3));
    // benchmark("caffe_yufacedetectnet-int8", ncnn::Mat(320, 240, 3));
    // benchmark("caffe_alexnet-int8", ncnn::Mat(227, 227, 3));
    // benchmark("caffe_googlenet-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_inception_v3-int8", ncnn::Mat(395, 395, 3));
    // benchmark("caffe_mnasnet-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_mobilenet_v1-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_mobilenet_v2-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_mobilenetv2_yolov3-int8", ncnn::Mat(320, 320, 3));
    // benchmark("caffe_resnet50-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_shufflenet_v1-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_shufflenet_v2-int8", ncnn::Mat(224, 224, 3));
    // benchmark("caffe_squeezenet_v1_1-int8", ncnn::Mat(227, 227, 3));
    // benchmark("caffe_squeezenet_ssd-int8", ncnn::Mat(300, 300, 3));
    // benchmark("caffe_vgg16-int8", ncnn::Mat(224, 224, 3));
    // benchmark("onnx_squeezenet-int8", ncnn::Mat(227, 227, 3));
    // benchmark("mxnet_alexnet-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_inception_v3-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_mobilenet_v1-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_mobilenet_v2_1_0-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_mobilenet_v2_0_25-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_resnet18_v2-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_resnet50-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_squeezenet_v1-int8", ncnn::Mat(224, 224, 3));
    // benchmark("mxnet_vgg16-int8", ncnn::Mat(224, 224, 3));

    //Convolution
	benchmark("conv_s1_k1x1x4x4_p0_d112x112x4", ncnn::Mat(112, 112, 4));
	benchmark("conv_s1_k2x2x8x8_p0_d56x56x8", ncnn::Mat(56, 56, 8));
	benchmark("conv_s1_k3x3x16x16_p1_d28x28x16", ncnn::Mat(28, 28, 16));
	benchmark("conv_s1_k4x4x32x32_p0_d14x14x32", ncnn::Mat(14, 14, 32));
	benchmark("conv_s1_k5x5x64x64_p2_d56x56x64", ncnn::Mat(56, 56, 64));
	benchmark("conv_s1_k7x7x12x12_p3_d112x112x12", ncnn::Mat(112, 112, 12));
	benchmark("conv_s1_k7x7x192x192_p3_d17x17x192", ncnn::Mat(17, 17, 192));
	benchmark("conv_s2_k1x1x32x3_p0_d224x224x3", ncnn::Mat(224, 224, 3));
	benchmark("conv_s2_k3x3x32x3_p1_d224x224x3", ncnn::Mat(224, 224, 3));
	benchmark("conv_s2_k5x5x32x3_p2_d224x224x3", ncnn::Mat(224, 224, 3));
	benchmark("conv_s2_k7x7x96x3_p3_d227x227x3", ncnn::Mat(227, 227, 3));

    //ConvolutionDepthWise group=2
	benchmark("conv_dw_s1_k3x3x32x32_p1_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("conv_dw_s1_k5x5x40x40_p2_d14x14x40", ncnn::Mat(14, 14, 40));
	benchmark("conv_dw_s2_k3x3x64x64_p1_d112x112x64", ncnn::Mat(112, 112, 64));
	benchmark("conv_dw_s2_k5x5x40x24_p2_d28x28x24", ncnn::Mat(28, 28, 24));

    //ConvolutionDepthWise group=inputc/outputc
	benchmark("conv_dw_s1_k3x3x32x32_p1_d112x112x32_g32", ncnn::Mat(112, 112, 32));
	benchmark("conv_dw_s1_k5x5x40x40_p2_d14x14x40_g40", ncnn::Mat(14, 14, 40));
	benchmark("conv_dw_s2_k3x3x64x64_p1_d112x112x64_g64", ncnn::Mat(112, 112, 64));
    // //invalid test case
	// // benchmark("conv_dw_s2_k5x5x40x24_p2_d28x28x24_g24", ncnn::Mat(28, 28, 24));

    //Deconvolution
	benchmark("deconv_s1_k3x3x32x32_p1_d40x40x32", ncnn::Mat(40, 40, 32));
	benchmark("deconv_s1_k4x4x16x16_p0_d80x80x16", ncnn::Mat(80, 80, 16));
	benchmark("deconv_s1_k5x5x12x12_p0_d80x80x12", ncnn::Mat(80, 80, 12));
	benchmark("deconv_s2_k2x2x64x4_p0_d160x160x4", ncnn::Mat(160, 160, 4));
	benchmark("deconv_s2_k3x3x8x8_p0_d80x80x8", ncnn::Mat(80, 80, 8));
	benchmark("deconv_s2_k4x4x4x4_p0_d80x80x4", ncnn::Mat(80, 80, 4));
	benchmark("deconv_s2_k5x5x12x12_p0_d80x80x12", ncnn::Mat(80, 80, 12));

    //DeconvolutionDepthWise group=2
    benchmark("deconv_dw_s1_k1x1x64x64_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k3x3x64x64_p1_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k4x4x64x64_p0_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k5x5x64x64_p2_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k1x1x64x64_p0_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k3x3x64x64_p1_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k4x4x64x64_p0_d40x40x64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k5x5x64x64_p2_d40x40x64", ncnn::Mat(40, 40, 64));

    //DeconvolutionDepthWise group=outputc
    benchmark("deconv_dw_s1_k1x1x64x64_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k3x3x64x64_p1_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k4x4x64x64_p0_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s1_k5x5x64x64_p2_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k1x1x64x64_p0_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k3x3x64x64_p1_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k4x4x64x64_p0_d40x40x64_g64", ncnn::Mat(40, 40, 64));
	benchmark("deconv_dw_s2_k5x5x64x64_p2_d40x40x64_g64", ncnn::Mat(40, 40, 64));

	//InnerProduct
	benchmark("fc_s1_n1000_d112x112x32", ncnn::Mat(112, 112, 32));
    benchmark("fc_s1_n1000_d7x7x1000", ncnn::Mat(7, 7, 1000));

	//Pooling MAX
	benchmark("pooling_max_s1_k2_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("pooling_max_s1_k3_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("pooling_max_s2_k2_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("pooling_max_s2_k3_d112x112x32", ncnn::Mat(112, 112, 32));

	//Pooling avg
	benchmark("pooling_avg_s1_k2_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("pooling_avg_s1_k3_d112x112x32", ncnn::Mat(112, 112, 32));
	benchmark("pooling_avg_s2_k2_d112x112x32", ncnn::Mat(112, 112, 32));
    benchmark("pooling_avg_s2_k3_d112x112x32", ncnn::Mat(112, 112, 32));

    //batchnorm
    benchmark("batchnorm_d19x19x1024", ncnn::Mat(19, 19, 1024));
	//hardsigmoid
	// benchmark("hardsigmoid_d112x112x4", ncnn::Mat(112, 112, 4));
	//hardswish
	benchmark("hardswish_d112x112x4", ncnn::Mat(112, 112, 4));
	//lrn
	benchmark("lrn_d55x55x96", ncnn::Mat(55, 55, 96));
	//relu
	benchmark("relu_d112x112x4", ncnn::Mat(112, 112, 4));
	//sigmoid
	benchmark("sigmoid_d112x112x4", ncnn::Mat(112, 112, 4));
	//softmax
	benchmark("softmax_d112x112x4", ncnn::Mat(112, 112, 4));
	//tanh
	benchmark("tanh_d112x112x4", ncnn::Mat(112, 112, 4));

	//eltwise
	benchmark("eltwise_max_d224x224x3", ncnn::Mat(224, 224, 3));
	benchmark("eltwise_prod_d224x224x3", ncnn::Mat(224, 224, 3));
	benchmark("eltwise_sum_d224x224x3", ncnn::Mat(224, 224, 3));
	//crop
	benchmark("crop_d112x112x3", ncnn::Mat(112, 112, 3));
	benchmark("crop_d224x224x3", ncnn::Mat(224, 224, 3));
	//concat
	benchmark("concat_d27x27x256", ncnn::Mat(27, 27, 256));
	//scale
	benchmark("scale_d112x112x4", ncnn::Mat(112, 112, 4));
	//selu
	benchmark("selu_d112x112x4", ncnn::Mat(112, 112, 4));

// 	fprintf(stderr, "======start performance test of models======\n");

// 	// run
//     fprintf(stderr, "======models(ncnn)======\n");
//     benchmark("squeezenet", ncnn::Mat(227, 227, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3));

//     benchmark("mobilenet", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3));

//     benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3));

// // #if NCNN_VULKAN
// //     if (!use_vulkan_compute)
// // #endif // NCNN_VULKAN
// //     benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3));

//     benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3));

//     benchmark("shufflenet", ncnn::Mat(224, 224, 3));

//     benchmark("mnasnet", ncnn::Mat(224, 224, 3));

//     benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3));

//     benchmark("googlenet", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("googlenet_int8", ncnn::Mat(224, 224, 3));

//     benchmark("resnet18", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("resnet18_int8", ncnn::Mat(224, 224, 3));

//     benchmark("alexnet", ncnn::Mat(227, 227, 3));

//     benchmark("vgg16", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("vgg16_int8", ncnn::Mat(224, 224, 3));

//     benchmark("resnet50", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("resnet50_int8", ncnn::Mat(224, 224, 3));

//     benchmark("squeezenet_ssd", ncnn::Mat(300, 300, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3));

//     benchmark("mobilenet_ssd", ncnn::Mat(300, 300, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3));

//     benchmark("mobilenet_yolo", ncnn::Mat(416, 416, 3));

//     benchmark("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3));


#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
