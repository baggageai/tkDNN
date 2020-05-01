#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "yolo_voc/layers/input.bin";
const char *c0_bin     = "yolo_voc/layers/c0.bin";
const char *c2_bin     = "yolo_voc/layers/c2.bin";
const char *c4_bin     = "yolo_voc/layers/c4.bin";
const char *c5_bin     = "yolo_voc/layers/c5.bin";
const char *c6_bin     = "yolo_voc/layers/c6.bin";
const char *c8_bin     = "yolo_voc/layers/c8.bin";
const char *c9_bin     = "yolo_voc/layers/c9.bin";
const char *c10_bin    = "yolo_voc/layers/c10.bin";
const char *c12_bin    = "yolo_voc/layers/c12.bin";
const char *c13_bin    = "yolo_voc/layers/c13.bin";
const char *c14_bin    = "yolo_voc/layers/c14.bin";
const char *c15_bin    = "yolo_voc/layers/c15.bin";
const char *c16_bin    = "yolo_voc/layers/c16.bin";
const char *c18_bin    = "yolo_voc/layers/c18.bin";
const char *c19_bin    = "yolo_voc/layers/c19.bin";
const char *c20_bin    = "yolo_voc/layers/c20.bin";
const char *c21_bin    = "yolo_voc/layers/c21.bin";
const char *c22_bin    = "yolo_voc/layers/c22.bin";
const char *c23_bin    = "yolo_voc/layers/c23.bin";
const char *c24_bin    = "yolo_voc/layers/c24.bin";
const char *c26_bin    = "yolo_voc/layers/c26.bin";
const char *c29_bin    = "yolo_voc/layers/c29.bin";
const char *c30_bin    = "yolo_voc/layers/c30.bin";
const char *g31_bin    = "yolo_voc/layers/g31.bin";
const char *output_bin = "yolo_voc/layers/output.bin";

int main() {

    downloadWeightsifDoNotExist(input_bin, "yolo_voc", "https://cloud.hipert.unimore.it/s/DJC5Fi2pEjfNDP9/download");

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 416, 416, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d     c0 (&net, 32, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tk::dnn::Activation a0 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p1 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c2 (&net, 64, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    tk::dnn::Activation a2 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p3 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c4 (&net, 128, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    tk::dnn::Activation a4 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c5 (&net, 64, 1, 1, 1, 1, 0, 0,   c5_bin, true);
    tk::dnn::Activation a5 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c6 (&net, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    tk::dnn::Activation a6 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p7 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c8 (&net, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    tk::dnn::Activation a8 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c9 (&net, 128, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    tk::dnn::Activation a9 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c10(&net, 256, 3, 3, 1, 1, 1, 1,  c10_bin, true);
    tk::dnn::Activation a10(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p11(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c12(&net, 512, 3, 3, 1, 1, 1, 1,  c12_bin, true);
    tk::dnn::Activation a12(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c13(&net, 256, 1, 1, 1, 1, 0, 0,  c13_bin, true);
    tk::dnn::Activation a13(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c14(&net, 512, 3, 3, 1, 1, 1, 1,  c14_bin, true);
    tk::dnn::Activation a14(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c15(&net, 256, 1, 1, 1, 1, 0, 0,  c15_bin, true);
    tk::dnn::Activation a15(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c16(&net, 512, 3, 3, 1, 1, 1, 1,  c16_bin, true); 
    tk::dnn::Activation a16(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p17(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c18(&net, 1024, 3, 3, 1, 1, 1, 1, c18_bin, true);
    tk::dnn::Activation a18(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c19(&net, 512, 1, 1, 1, 1, 0, 0,  c19_bin, true);
    tk::dnn::Activation a19(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c20(&net, 1024, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tk::dnn::Activation a20(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c21(&net, 512, 1, 1, 1, 1, 0, 0,  c21_bin, true);
    tk::dnn::Activation a21(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c22(&net, 1024, 3, 3, 1, 1, 1, 1, c22_bin, true);
    tk::dnn::Activation a22(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c23(&net, 1024, 3, 3, 1, 1, 1, 1, c23_bin, true);
    tk::dnn::Activation a23(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c24(&net, 1024, 3, 3, 1, 1, 1, 1, c24_bin, true);
    tk::dnn::Activation a24(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *m25_layers[1] = { &a16 };
    tk::dnn::Route      m25(&net, m25_layers, 1);
    tk::dnn::Conv2d     c26(&net, 64, 1, 1, 1, 1, 0, 0,   c26_bin, true);
    tk::dnn::Activation a26(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Reorg      r27(&net, 2);

    tk::dnn::Layer *m28_layers[2] = { &r27, &a24 };
    tk::dnn::Route      m28(&net, m28_layers, 2);

    tk::dnn::Conv2d     c29(&net, 1024, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tk::dnn::Activation a29(&net, tk::dnn::ACTIVATION_LEAKY);   
    tk::dnn::Conv2d     c30(&net, 125, 1, 1, 1, 1, 0, 0,  c30_bin, false);
    tk::dnn::Region     g31(&net, 20, 4, 5);
    
    tk::dnn::RegionInterpret rI(dim, g31.output_dim, 20, 4, 5, 0.6f, g31_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("yolo_voc"));

    dnnType *out_data, *out_data2; // cudnn output, tensorRT output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
    
    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        out_data2 = netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; 
    int ret_cudnn = checkResult(out_dim, out_data, out) == 0 ? 0: ERROR_CUDNN;
    std::cout<<"TRT   vs correct"; 
    int ret_tensorrt = checkResult(out_dim, out_data2, out) == 0 ? 0 : ERROR_TENSORRT;
    std::cout<<"CUDNN vs TRT    "; 
    int ret_cudnn_tensorrt = checkResult(out_dim, out_data, out_data2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    // std::cout<<"\n\nDetected objects: \n";
    // dnnType *output_h = new dnnType[rI.output_dim.tot()];
    // checkCuda(cudaMemcpy(output_h, out_data2, 
    //           rI.output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost));
    // rI.interpretData(output_h);
    // rI.showImageResult(input_h);

    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
