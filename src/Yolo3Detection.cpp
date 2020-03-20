#include "Yolo3Detection.h"


namespace tk { namespace dnn {

bool Yolo3Detection::init(const std::string& tensor_path, const int n_classes) {

    //convert network to tensorRT
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );

    if(netRT->pluginFactory->n_yolos < 2 ) {
        FatalError("this is not yolo3");
    }

    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        YoloRT *yRT = netRT->pluginFactory->yolos[i];
        classes = yRT->classes;
        num = yRT->num;
        nMasks = yRT->n_masks;

        // make a yolo layer for interpret predictions
        yolo[i] = new tk::dnn::Yolo(nullptr, classes, nMasks, ""); // yolo without input and bias
        yolo[i]->mask_h = new dnnType[nMasks];
        yolo[i]->bias_h = new dnnType[num*nMasks*2];
        memcpy(yolo[i]->mask_h, yRT->mask, sizeof(dnnType)*nMasks);
        memcpy(yolo[i]->bias_h, yRT->bias, sizeof(dnnType)*num*nMasks*2);
        yolo[i]->input_dim = yolo[i]->output_dim = tk::dnn::dataDim_t(1, yRT->c, yRT->h, yRT->w);
        yolo[i]->classesNames = yRT->classesNames;
    }

    dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
#ifndef OPENCV_CUDA
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*netRT->input_dim.tot()));
#endif
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot()));

    // class colors precompute    
    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    return true;
} 

void Yolo3Detection::preprocess(cv::Mat &frame)
{
#ifdef OPENCV_CUDA
    cv::cuda::GpuMat orig_img, img_resized;
    orig_img = cv::cuda::GpuMat(frame);
    cv::cuda::resize(orig_img, img_resized, cv::Size(netRT->input_dim.w, netRT->input_dim.h));

    img_resized.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::cuda::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        std::cout<<"copio il channel"<<i<<std::endl;
        int idx = i*imagePreproc.rows*imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        checkCuda( cudaMemcpy((void*)&input_d[idx], (void*)bgr[ch].data, imagePreproc.rows*imagePreproc.cols*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    }
#else
    cv::resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
    frame.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int idx = i*imagePreproc.rows*imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*)&input[idx], (void*)bgr[ch].data, imagePreproc.rows*imagePreproc.cols*sizeof(dnnType));
    }
    checkCuda(cudaMemcpyAsync(input_d, input, netRT->input_dim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
#endif
}

void Yolo3Detection::update(cv::Mat &frame)
{
    TIMER_START
    if(!frame.data) {
        std::cout<<"YOLO: NO IMAGE DATA\n";
        return;
    }   

    originalSize = frame.size();
    preprocess(frame);

    //do inference
    tk::dnn::dataDim_t dim = netRT->input_dim;
    
    printCenteredTitle(" TENSORRT inference ", '=', 30); 
    {
        dim.print();
        TIMER_START
        netRT->infer(dim, input_d);
        TIMER_STOP
        dim.print();
    }

    //get yolo outputs
    dnnType *rt_out[netRT->pluginFactory->n_yolos]; 
    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        rt_out[i] = (dnnType*)netRT->buffersRT[i+1];
    }

    postprocess(rt_out, netRT->pluginFactory->n_yolos);
        
    TIMER_STOP
    stats.push_back(t_ns);
}

void Yolo3Detection::postprocess(dnnType **rt_out, const int n_out)
{
    float x_ratio =  float(originalSize.width) / float(netRT->input_dim.w);
    float y_ratio =  float(originalSize.height) / float(netRT->input_dim.h);

    std::cout<<"RATIO:"<<x_ratio<<" "<<y_ratio<<std::endl;

    // compute dets
    nDets = 0;
    for(int i=0; i<n_out; i++) {
        yolo[i]->dstData = rt_out[i];
        yolo[i]->computeDetections(dets, nDets, netRT->input_dim.w, netRT->input_dim.h, confThreshold);
    }
    tk::dnn::Yolo::mergeDetections(dets, nDets, classes);

    // fill detected
    detected.clear();
    for(int j=0; j<nDets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        int x0   = (b.x-b.w/2.);
        int x1   = (b.x+b.w/2.);
        int y0   = (b.y-b.h/2.);
        int y1   = (b.y+b.h/2.);
        int obj_class = -1;
        float prob = 0;
        for(int c=0; c<classes; c++) {
            if(dets[j].prob[c] >= confThreshold) {
                obj_class = c;
                prob = dets[j].prob[c];
            }
        }

        if(obj_class >= 0) {
            // convert to image coords
            x0 = x_ratio*x0;
            x1 = x_ratio*x1;
            y0 = y_ratio*y0;
            y1 = y_ratio*y1;
              
            tk::dnn::box res;
            res.cl = obj_class;
            res.prob = prob;
            res.x = x0;
            res.y = y0;
            res.w = x1 - x0;
            res.h = y1 - y0;
            detected.push_back(res);
        }
    }
    std::cout<<"N detections: "<<detected.size()<<std::endl;
}

cv::Mat Yolo3Detection::draw(cv::Mat &frame) 
{
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   
    // draw dets
    for(int i=0; i<detected.size(); i++) {
        b           = detected[i];
        x0   		= b.x;
        x1   		= b.x + b.w;
        y0   		= b.y;
        y1   		= b.y + b.h;
        det_class 	= getYoloLayer()->classesNames[b.cl];

        // draw rectangle
        cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

        // draw label
        cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
        cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
    }
    return frame;
}

tk::dnn::Yolo* Yolo3Detection::getYoloLayer(int n) 
{
    if(n<3)
        return yolo[n];
    else 
        return nullptr;
}

}}
