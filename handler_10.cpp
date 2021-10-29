#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#ifdef __linux__
#include <unistd.h>
#endif
#include "stb_image.h"
#include <mutex>
#include "utils.h"
#include "baggageDetect.hpp"
#include "handler.h"
#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>
#include <string>
#include <fstream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Yolo3Detection.h"
//#include "CenternetDetection.h"
//#include "MobilenetDetection.h"
#include "evaluation.h"
#include "tkdnn.h"
#include <chrono>
#include <cstdint>
#include <iostream>
using namespace std;
using namespace cv;
#include <fstream>
#include <iostream>
#include <string>
#include "image.h"
void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}
int check_mistakes = 0;
image load_image_file(unsigned char *image_data, int channels, int antilog, int gray, int width, int height)
{
    int w, h, c;
    unsigned char *data = image_data;
    w = width;
    h = height;
    c = channels;

    if (!image_data) {
        if (check_mistakes) getchar();
        return make_image(10, 10, 3);
        
    }
    if (channels) c = channels;

    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                (im).data[dst_index] = (float)image_data[src_index]/255.;
            }
        }
    }
    //free(data);
    return im;
}
cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;
 
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}   
    std::vector<std::string> classesNames;
    image im;
    cv::Mat frame;
    cv::Mat gray;
    int h=0;
    int w=0;
    int channels;
    const char *config_filename = "config/config.yaml";
    const char * net1="config/battery_e_cigratte.rt";
    const char * net2="config/cigar_cigratte.rt";
    const char * net3="config/jwellery_watches.rt"; 
    const char * net4="config/battery_e_cigratte.rt";
    const char * net5="config/cigar_cigratte.rt";
    const char * net6="config/jwellery_watches.rt";
    const char * net7="config/battery_e_cigratte.rt";
    const char * net8="config/cigar_cigratte.rt";
    const char * net9="config/jwellery_watches.rt";
    const char * net10="config/battery_e_cigratte.rt";

    int classes1 , classes2 , classes3 , classes4 , classes5,len;
    char * img_data;

    string ustring;
    float conf_thresh1 , conf_thresh2 , conf_thresh3 , conf_thresh4 , conf_thresh5;
    tk::dnn::Yolo3Detection yolo1;
    tk::dnn::DetectionNN *detNN1;
    tk::dnn::Yolo3Detection yolo2;
    tk::dnn::DetectionNN *detNN2;
    tk::dnn::Yolo3Detection yolo3;
    tk::dnn::DetectionNN *detNN3;
    tk::dnn::Yolo3Detection yolo4;
    tk::dnn::DetectionNN *detNN4;
    tk::dnn::Yolo3Detection yolo5;
    tk::dnn::DetectionNN *detNN5;
    tk::dnn::Yolo3Detection yolo6;
    tk::dnn::DetectionNN *detNN6;
    tk::dnn::Yolo3Detection yolo7;
    tk::dnn::DetectionNN *detNN7;
    tk::dnn::Yolo3Detection yolo8;
    tk::dnn::DetectionNN *detNN8;
    tk::dnn::Yolo3Detection yolo9;
    tk::dnn::DetectionNN *detNN9;
    tk::dnn::Yolo3Detection yolo10;
    tk::dnn::DetectionNN *detNN10;
    unsigned char * sockData;
    
    std::vector<cv::Mat> batch_frames;
    std::vector<cv::Mat> batch_dnn_input;
    std::vector<std::string> classesNames1;
    std::vector<std::string> classesNames2;
    std::vector<std::string> classesNames3;
    std::vector<std::string> classesNames4;
    std::vector<std::string> classesNames5;
    std::vector<std::string> classesNames6;
    std::vector<std::string> classesNames7;
    std::vector<std::string> classesNames8;
    std::vector<std::string> classesNames9;
     std::vector<std::string> classesNames10;
    std::vector<tk::dnn::Frame> images;
    std::vector<tk::dnn::box> detected_bbox1;
    std::vector<tk::dnn::box> detected_bbox2;
    std::vector<tk::dnn::box> detected_bbox3;
    std::vector<tk::dnn::box> detected_bbox4;
    std::vector<tk::dnn::box> detected_bbox5;
    std::vector<tk::dnn::box> detected_bbox6;
    std::vector<tk::dnn::box> detected_bbox7;
    std::vector<tk::dnn::box> detected_bbox8;
    std::vector<tk::dnn::box> detected_bbox9;
    std::vector<tk::dnn::box> detected_bbox10;
    //read parametersi
    handler::handler(utility::string_t url):m_listener(url)
{
    m_listener.support(methods::POST, bind(&handler::handle_post, this, placeholders::_1));

}

string name_from_path(string path)
{
    return path.substr(path.find_last_of("/\\")+1);
}
    void handler::init_bag(){tk::dnn::readmAPParams(config_filename, classes1,conf_thresh1, classes2,conf_thresh2
    , classes3,conf_thresh3, classes4,conf_thresh4,classes5,conf_thresh5);

    detNN1 = &yolo1;
    detNN1->init(net1, classes1, 1, conf_thresh1);
    classesNames1=detNN1-> classesNames;
    detNN2 = &yolo2;
    detNN2->init(net2, classes2, 1, conf_thresh2);
    classesNames2=detNN2-> classesNames;
    detNN3 = &yolo3;
    detNN3->init(net3, classes3, 1, conf_thresh3);
    classesNames3=detNN3-> classesNames;
    detNN3 = &yolo3;
    detNN3->init(net3, classes3, 1, conf_thresh3);
    classesNames3=detNN3-> classesNames;
    detNN4 = &yolo4;
    detNN4->init(net4, classes1, 1, conf_thresh1);
    classesNames4=detNN4-> classesNames;
    detNN5 = &yolo5;
    detNN5->init(net5, classes2, 1, conf_thresh2);
    classesNames5=detNN5-> classesNames;
    detNN6 = &yolo6;
    detNN6->init(net6, classes3, 1, conf_thresh3);
    classesNames6=detNN6-> classesNames;
    detNN7 = &yolo7;
    detNN7->init(net7, classes1, 1, conf_thresh1);
    classesNames7=detNN7-> classesNames;
    detNN8 = &yolo8;
    detNN8->init(net8, classes2, 1, conf_thresh2);
    classesNames8=detNN8-> classesNames;
    detNN9 = &yolo9;
    detNN9->init(net9, classes3, 1, conf_thresh3);
    classesNames9=detNN9-> classesNames;
    detNN10 = &yolo10;
    detNN10->init(net10, classes1, 1, conf_thresh1);
    classesNames10=detNN10-> classesNames;

    return;}


    void handler::handle_post(http_request request){
    BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << request.to_string();

    map<utility::string_t, utility::string_t> http_get_vars = uri::split_query(request.request_uri().query());
    map<utility::string_t, utility::string_t>::iterator it = http_get_vars.find("name");
   // int len;
    if(it == http_get_vars.end())
    {
        BOOST_LOG_TRIVIAL(error) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Image name not passed in query.";
        request.reply(status_codes::UnprocessableEntity,"Please pass image name in the query.");
        return;
    }https://github.com/baggageai/baggageai-code-one.git
   // std::cout<<http_get_vars["name"]<<"\n";
    string image_name = (string)http_get_vars["name"];
   // string ustring;
    request.extract_vector().then([image_name, &ustring, &len](vector<unsigned char> v) {
                ustring = {v.begin(),v.end()};
                len = ustring.size();
            }).wait();
    unsigned char *idata;
try {    //printSize(ustring);
    BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Detection Started";
    sockData = (unsigned char *)ustring.c_str();
    idata = stbi_load_from_memory(sockData, len, &w, &h, &channels, 0);
    im = load_image_file(idata, channels, 0, 0, w, h);
    batch_dnn_input.clear();
    //batch_frames.clear();
    gray=image_to_mat(im);
//    free(im);
    cv::Mat in[] = {gray, gray,gray};
    cv::merge(in, 3, frame);
    batch_dnn_input.push_back(frame.clone());
    json::value response;
    vector<json::value> jsonArray;
   
    detected_bbox1.clear();
    detNN1->update(batch_dnn_input,1);
    detected_bbox1 = detNN1->detected;
    
    for(auto d1:detected_bbox1){	
	    json::value detection;
        BOOST_LOG_TRIVIAL(info)<<"model-1"<<" "<< d1.cl << " "<< d1.prob << " "<< d1.x << " "<< d1.y << " "<< d1.w << " "<< d1.h <<"\n";
        detection["label"] = json::value::string(classesNames1[d1.cl]);
        detection["x"] = json::value::number(d1.x);
        detection["y"] = json::value::number(d1.y);
        detection["w"] = json::value::number(d1.w);
        detection["h"] = json::value::number(d1.h);
        detection["prob"] = json::value::number(d1.prob);
        jsonArray.push_back(detection);
}
     detected_bbox2.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN2->update(batch_dnn_input,1);
   // std::cout<<batch_dnn_input[0].size()<<" testing3\n";
    detected_bbox2 = detNN2->detected;

    for(auto d2:detected_bbox2){	
        BOOST_LOG_TRIVIAL(info)<<"model-2:"<<" "<< d2.cl << " "<< d2.prob << " "<< d2.x << " "<< d2.y << " "<< d2.w << " "<< d2.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames2[d2.cl]);
        detection["x"] = json::value::number(d2.x);
        detection["y"] = json::value::number(d2.y);
        detection["w"] = json::value::number(d2.w);
        detection["h"] = json::value::number(d2.h);
        detection["prob"] = json::value::number(d2.prob);
        jsonArray.push_back(detection);
}
     detected_bbox3.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN3->update(batch_dnn_input,1);
    detected_bbox3 = detNN3->detected;
    for(auto d3:detected_bbox3){	
        BOOST_LOG_TRIVIAL(info)<< "model-3:"<<" "<<d3.cl << " "<< d3.prob << " "<< d3.x << " "<< d3.y << " "<< d3.w << " "<< d3.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames3[d3.cl]);
        detection["x"] = json::value::number(d3.x);
        detection["y"] = json::value::number(d3.y);
        detection["w"] = json::value::number(d3.w);
        detection["h"] = json::value::number(d3.h);
        detection["prob"] = json::value::number(d3.prob);
        jsonArray.push_back(detection);
}       
     detected_bbox4.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN4->update(batch_dnn_input,1);
    detected_bbox4 = detNN4->detected;
    for(auto d4:detected_bbox4){	
        BOOST_LOG_TRIVIAL(info)<< "model-4:"<<" "<<d4.cl << " "<< d4.prob << " "<< d4.x << " "<< d4.y << " "<< d4.w << " "<< d4.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames4[d4.cl]);
        detection["x"] = json::value::number(d4.x);
        detection["y"] = json::value::number(d4.y);
        detection["w"] = json::value::number(d4.w);
        detection["h"] = json::value::number(d4.h);
        detection["prob"] = json::value::number(d4.prob);
        jsonArray.push_back(detection);
}       

         detected_bbox5.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN5->update(batch_dnn_input,1);
    detected_bbox5 = detNN5->detected;
    for(auto d5:detected_bbox5){	
        BOOST_LOG_TRIVIAL(info)<< "model-5:"<<" "<<d5.cl << " "<< d5.prob << " "<< d5.x << " "<< d5.y << " "<< d5.w << " "<< d5.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames5[d5.cl]);
        detection["x"] = json::value::number(d5.x);
        detection["y"] = json::value::number(d5.y);
        detection["w"] = json::value::number(d5.w);
        detection["h"] = json::value::number(d5.h);
        detection["prob"] = json::value::number(d5.prob);
        jsonArray.push_back(detection);
}       
         detected_bbox6.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN6->update(batch_dnn_input,1);
    detected_bbox6 = detNN6->detected;
    for(auto d6:detected_bbox6){	
        BOOST_LOG_TRIVIAL(info)<< "model-6:"<<" "<<d6.cl << " "<< d6.prob << " "<< d6.x << " "<< d6.y << " "<< d6.w << " "<< d6.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames6[d6.cl]);
        detection["x"] = json::value::number(d6.x);
        detection["y"] = json::value::number(d6.y);
        detection["w"] = json::value::number(d6.w);
        detection["h"] = json::value::number(d6.h);
        detection["prob"] = json::value::number(d6.prob);
        jsonArray.push_back(detection);
}       
         detected_bbox7.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN7->update(batch_dnn_input,1);
    detected_bbox7 = detNN7->detected;
    for(auto d7:detected_bbox7){	
        BOOST_LOG_TRIVIAL(info)<< "model-7:"<<" "<<d7.cl << " "<< d7.prob << " "<< d7.x << " "<< d7.y << " "<< d7.w << " "<< d7.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames7[d7.cl]);
        detection["x"] = json::value::number(d7.x);
        detection["y"] = json::value::number(d7.y);
        detection["w"] = json::value::number(d7.w);
        detection["h"] = json::value::number(d7.h);
        detection["prob"] = json::value::number(d7.prob);
        jsonArray.push_back(detection);
}       
         detected_bbox8.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN8->update(batch_dnn_input,1);
    detected_bbox8 = detNN8->detected;
    for(auto d8:detected_bbox8){	
        BOOST_LOG_TRIVIAL(info)<< "model-8:"<<" "<<d8.cl << " "<< d8.prob << " "<< d8.x << " "<< d8.y << " "<< d8.w << " "<< d8.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames8[d8.cl]);
        detection["x"] = json::value::number(d8.x);
        detection["y"] = json::value::number(d8.y);
        detection["w"] = json::value::number(d8.w);
        detection["h"] = json::value::number(d8.h);
        detection["prob"] = json::value::number(d8.prob);
        jsonArray.push_back(detection);
}       
         detected_bbox9.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN9->update(batch_dnn_input,1);
    detected_bbox9 = detNN9->detected;
    for(auto d9:detected_bbox9){	
        BOOST_LOG_TRIVIAL(info)<< "model-9:"<<" "<<d9.cl << " "<< d9.prob << " "<< d9.x << " "<< d9.y << " "<< d9.w << " "<< d9.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames9[d9.cl]);
        detection["x"] = json::value::number(d9.x);
        detection["y"] = json::value::number(d9.y);
        detection["w"] = json::value::number(d9.w);
        detection["h"] = json::value::number(d9.h);
        detection["prob"] = json::value::number(d9.prob);
        jsonArray.push_back(detection);
}       
         detected_bbox10.clear();
    batch_dnn_input.clear();
    batch_dnn_input.push_back(frame.clone());
    detNN10->update(batch_dnn_input,1);
    detected_bbox10 = detNN10->detected;
    for(auto d10:detected_bbox10){	
        BOOST_LOG_TRIVIAL(info)<< "model-10:"<<" "<<d10.cl << " "<< d10.prob << " "<< d10.x << " "<< d10.y << " "<< d10.w << " "<< d10.h <<"\n";
	    json::value detection;
        detection["label"] = json::value::string(classesNames10[d10.cl]);
        detection["x"] = json::value::number(d10.x);
        detection["y"] = json::value::number(d10.y);
        detection["w"] = json::value::number(d10.w);
        detection["h"] = json::value::number(d10.h);
        detection["prob"] = json::value::number(d10.prob);
        jsonArray.push_back(detection);
}       
        

            response["detections"] = json::value::array(jsonArray);   //JSON Response
//        free(jsonArray);
        request.reply(status_codes::OK,response.serialize());
//        free(detected_bbox);
        BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Detection Completed and Response sent";
    }
    catch (exception const& e) {
        BOOST_LOG_TRIVIAL(error) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << e.what();
        request.reply(status_codes::BadRequest, e.what());
    }
  //  std::cout << timeSinceEpochMillisec() << std::endl;
    free(idata);
    free_image(im);
    return ;
    }

