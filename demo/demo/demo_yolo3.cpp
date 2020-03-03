#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Yolo3Detection.h"

bool 			gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    char *net = "yolo3_berkeley.rt";
    if(argc > 1)
        net = argv[1]; 
    char *input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";


    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    
    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            break;
        }  
 
        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        frame = yolo.draw(frame);
        
        cv::imshow("detection", frame);
        cv::waitKey(1);
        if(SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   


    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(yolo.stats.begin(), yolo.stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(yolo.stats.begin(), yolo.stats.end())<<" ms\n";    
    double mean = 0; for(int i=0; i<yolo.stats.size(); i++) mean += yolo.stats[i]; mean /= yolo.stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;    
    return 0;
}

