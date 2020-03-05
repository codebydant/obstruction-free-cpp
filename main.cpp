//
//  main.cpp
//  opencv-test
//
//  Created by XU BINBIN on 4/4/17.
//  Copyright © 2017 XU BINBIN. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

int pyramid_level = 2;
size_t frameNumber;
size_t reference_number;

void colorFlow(cv::Mat flow, std::string figName);
void drawOptFlowMap(const cv::Mat &flow, cv::Mat &cflowmap, int step, double, const cv::Scalar &color);
cv::Mat indexToMask(cv::Mat indexMat, int rows, int cols);

// use Homography to filter outliers in the flow
cv::Mat flowHomography(cv::Mat edges, cv::Mat flow, int ransacThre);

// interpolate from sparse edgeflow to dense optical flow
cv::Mat sparse_int_dense(cv::Mat im1, cv::Mat im2, cv::Mat im1_edges, cv::Mat sparseFlow);

cv::Mat imgWarpFlow(cv::Mat im1, cv::Mat flow);

// add flow src_flow + add_flow=>obj_flow
cv::Mat addFlow(cv::Mat src_flow, cv::Mat add_flow);

void initila_motion_decompose(cv::Mat im1, cv::Mat im2, cv::Mat &back_denseFlow, cv::Mat &fore_denseFlow,
                              int back_ransacThre, int fore_ransacThre);
// motion fields initialization
// direct matching to the reference frame
void motion_initiliazor_direct(const std::vector<cv::Mat> &video_input, std::vector<cv::Mat> &back_flowfields,
                               std::vector<cv::Mat> &fore_flowfields, std::vector<cv::Mat> &warpedToReference);
// matching between neighbouring frames and warping to the reference frame
void motion_initiliazor_iterative(const std::vector<cv::Mat> &video_input, std::vector<cv::Mat> &back_flowfields,
                                  std::vector<cv::Mat> &fore_flowfields, std::vector<cv::Mat> &warpedToReference);
// irls motion decomposition
void mot_decom_irls(const std::vector<cv::Mat> &input_sequence, cv::Mat &backgd_comp, cv::Mat &obstruc_comp,
                    cv::Mat &alpha_map, std::vector<cv::Mat> back_flowfields, std::vector<cv::Mat> fore_flowfields,
                    int nOuterFPIterations);

cv::Mat Laplac(const cv::Mat &input);
cv::Mat imshow32F(cv::Mat A);

void motDecomIrlsWeight(const std::vector<cv::Mat> &input_sequence, const cv::Mat &backgd_comp,
                        const cv::Mat &obstruc_comp, cv::Mat &alpha_map, const std::vector<cv::Mat> &back_flowfields,
                        const std::vector<cv::Mat> &fore_flowfields, std::vector<float> &omega_1,
                        std::vector<float> &omega_2, std::vector<float> &omega_3);

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

int main(int argc, const char *argv[]) {

  // some parameters

  if (argc < 2) {
    std::cout << "Usage: <executable> <images path>" << std::endl;
    std::exit(-1);
  }

  std::cout << "\n*************************************" << std::endl;
  std::cout << "*** OBSTRUCTION FREE PHOTOGRAPHY *** " << std::endl;
  std::cout << "*************************************\n" << std::endl;

  std::vector<cv::Mat> back_flowfields;
  std::vector<cv::Mat> fore_flowfields;
  std::vector<cv::Mat> warpedToReference;
  cv::Mat alpha_map;
  cv::Mat background;
  cv::Mat foregrond;

  ////////////////////input image sequences//////////////////////
  std::vector<cv::Mat> video_input;
  std::vector<cv::Mat> video_coarseLeve;
  cv::Mat referFrame;
  cv::Mat currentFrame;

  const int x_size = 1000;
  const int y_size = 400;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "-> Getting images - " << std::flush;

  cv::VideoCapture capture(argv[1]);
  frameNumber = capture.get(cv::CAP_PROP_FRAME_COUNT);
  reference_number = (frameNumber - 1) / 2;

  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    capture >> currentFrame;
    if (frame_i == reference_number) {
      referFrame = currentFrame.clone();
    }
    cv::Mat current_frame_resized;
    // cv::resize(currentFrame, current_frame_resized, cv::Size(640, 480));
    cv::Mat current_frame_resized_blur;
    cv::GaussianBlur(currentFrame, current_frame_resized_blur, cv::Size(3, 3), 0, 0);
    video_input.push_back(current_frame_resized_blur.clone());
    // imshow("inputVideo", currentFrame);
    // waitKey(10);
  }
  // cv::namedWindow("referFrame", cv::WINDOW_NORMAL);
  // cv::resizeWindow("referFrame", x_size, y_size);
  // imshow("referFrame", referFrame);
  std::cout << "[OK]" << std::endl;
  std::cout << "-> Number of images in vector: [" << video_input.size() << "]" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "-> RGB to grey scale - " << std::flush;
  std::cout << "[OK]" << std::endl;
  std::cout << "-> Downsampling grey images - " << std::flush;

  /////////construct image pyramids//////
  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    cv::Mat temp, temp_gray;
    temp = video_input[frame_i].clone();
    cv::cvtColor(temp, temp_gray, cv::COLOR_RGB2GRAY);
    for (int i = 0; i < pyramid_level; i++) {
      cv::pyrDown(temp_gray, temp_gray);
    }
    video_coarseLeve.push_back(temp_gray.clone());
  }

  std::cout << "[OK]" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  std::cout << "-> Motion initiliazor direct" << std::endl;

  /////////////////initialization->motion fields for back/foreground layers///////////
  motion_initiliazor_direct(video_coarseLeve, back_flowfields, fore_flowfields, warpedToReference);
  // motion_initiliazor_iterative(video_coarseLeve, back_flowfields, fore_flowfields, warpedToReference);
  ////////////show warped image frames/////////////
  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    char windowName[10] = "yeah";
    // std::cout << "warped " + std::to_string(frame_i) << std::endl;
    // cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    // cv::resizeWindow(windowName, x_size, y_size);
    // imshow(windowName, warpedToReference[frame_i]);
    // char flowwindow[10];
    // sprintf(flowwindow, "flow %d", frame_i);
    // colorFlow(back_flowfields[frame_i], flowwindow);
  }

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "-> Initialization" << std::endl;

  //////////////////////////Initialization/////////////////

  /////// opaque occlusion/////////////
  cv::Mat sum =
      cv::Mat::zeros(warpedToReference[reference_number].rows, warpedToReference[reference_number].cols, CV_32F);
  cv::Mat temp, background_temp;
  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    warpedToReference[frame_i].convertTo(temp, CV_32F);
    sum += temp;
  }

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "-> Opaque initial background" << std::endl;

  background_temp = sum / frameNumber;
  background_temp.convertTo(background, CV_8UC1);
  // cv::namedWindow("opaque initial background", cv::WINDOW_NORMAL);
  // cv::resizeWindow("opaque initial background", x_size, y_size);
  // imshow("opaque initial background", background);

  warpedToReference[reference_number].convertTo(temp, CV_32F);
  cv::Mat difference;
  difference = abs(background - warpedToReference[reference_number]);
  threshold(difference, alpha_map, 25.5, 255, cv::THRESH_BINARY_INV);
  // cv::namedWindow("alpha map", cv::WINDOW_NORMAL);
  // cv::resizeWindow("alpha map", x_size, y_size);
  // imshow("alpha map", alpha_map);
  // cout<<alpha_map<<endl;

  foregrond = warpedToReference[reference_number] - background;
  // cv::namedWindow("foreground", cv::WINDOW_NORMAL);
  // cv::resizeWindow("foreground", x_size, y_size);
  // imshow("foreground", foregrond);

  //            ////////  reflection pane///////////////////
  //    background=warpedToReference[reference_number];
  //    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
  //        background=min(background,warpedToReference[frame_i]);
  //    }
  //    imshow("reflection initial background", background);

  ////////////////////IRLS decomposition/////////////////
  cv::Mat lapForeground;
  cv::Mat lapBackground;

  // std::cout << "-> Background type: " << background.type() << std::endl;
  // std::cout << "-> Background type: " << (int)background.at<unsigned char>(5, 5) << std::endl;
  // background.convertTo(test,CV_32F, 1.0/255.0f);

  // Mat1f test(background);

  lapForeground = Laplac(foregrond);
  lapBackground = Laplac(background);

  // cv::namedWindow("laplcian-backgounrd", cv::WINDOW_NORMAL);
  // cv::resizeWindow("laplcian-backgounrd", x_size, y_size);
  // imshow("laplcian-backgounrd", lapBackground);
  // cout<<lapBackground<<endl;
  // std::cout << "-> Background type: " << background.type() << std::endl;
  // std::cout << "-> Background type: " << lapBackground.type() << std::endl;

  cv::Mat lap1;
  lapBackground.convertTo(lap1, CV_8U);
  // cv::namedWindow("lap2", cv::WINDOW_NORMAL);
  // cv::resizeWindow("lap2", x_size, y_size);
  // imshow("lap2", lap1);

  // cv::namedWindow("laplcian-foregrond", cv::WINDOW_NORMAL);
  // cv::resizeWindow("laplcian-foregrond", x_size, y_size);
  // imshow("laplcian-foregrond", lapForeground);
  // cout<<lap1<<endl;

  // int nOuterFPIterations=5;
  // int nInnerFPIterations=3;
  // for(int ocount=0; ocount<nOuterFPIterations; ocount++){
  //        for(int icount =0; icount< nInnerFPIterations; icount++){
  //        }
  //}

  std::vector<float> omega_1;
  std::vector<float> omega_2;
  std::vector<float> omega_3;

  motDecomIrlsWeight(video_coarseLeve, background, foregrond, alpha_map, back_flowfields, fore_flowfields, omega_1,
                     omega_2, omega_3);

  motDecomIrlsWeight(video_coarseLeve, background, foregrond, alpha_map, back_flowfields, fore_flowfields, omega_1,
                     omega_2, omega_3);

  ////////////////////IRLS motion estimation/////////////////

  /////////////////////MRF////////////////////////////////////////////////

  // int patch = 5; // patch size (2*patch+1)^2
  // // remove edges near the image borders
  // for (int y = 0; y < im1_edge.rows; y++) {
  //   for (int x = 0; x < im1_edge.cols; x++) {
  //     if (y < patch || x < patch || x >= (im1_edge.cols - patch) || y >= (im1_edge.rows - patch)) {
  //       im1_edge.at<uchar>(y, x) = 0;
  //       im2_edge.at<uchar>(y, x) = 0;
  //     }
  //   }
  // }

  // //#label=#edge in image2
  // Mat label_Locations;
  // findNonZero(im2_edge, label_Locations);
  // int nL = label_Locations.total();

  // // data-term
  // MRF::CostVal *cData = NULL;
  // computeCost(im1_grey, im2_grey, im1_edge, im2_edge, cData, patch, nL);
  // DataCost *dcost = new DataCost(cData);

  // SmoothnessCost *scost;
  // MRF::CostVal *hCue = NULL, *vCue = NULL;
  // if (gradThresh > 0) {
  //   computeCues(im1, hCue, vCue, gradThresh, gradPenalty);
  //   scost = new SmoothnessCost(smoothexp, smoothmax, lambda, hCue, vCue);
  // } else {
  //   scost = new SmoothnessCost(smoothexp, smoothmax, lambda);
  // }
  // EnergyFunction *energy = new EnergyFunction(dcost, scost);
  std::cout << "\nVisualizing results...\n=============================================\n";

  // std::string ty = type2str(foregrond.type());
  // printf("Matrix: %s %dx%d \n", ty.c_str(), foregrond.cols, foregrond.rows);

  cv::Mat out;
  // foregrond.convertTo(foregrond, CV_8UC3); // CV_8UC1
  // referFrame.convertTo(referFrame, CV_8UC3);
  // out.convertTo(out, CV_8UC1);
  cv::Size s1 = foregrond.size();
  cv::Size s2 = referFrame.size();

  cv::Mat foregrond_3_CH;
  cv::Mat referFrame_3_CH_resized;
  cv::resize(referFrame, referFrame_3_CH_resized, cv::Size(s1.width, s1.height));

  // std::cout << "foreground type:" << type2str(foregrond.type()) << std::endl;
  // std::cout << "foreground channels:" << foregrond.channels() << std::endl;
  // std::cout << "foreground size:" << foregrond.size() << std::endl;

  cv::cvtColor(foregrond, foregrond_3_CH, cv::COLOR_GRAY2BGR);

  // std::cout << "foreground type:" << type2str(foregrond_3_CH.type()) << std::endl;
  // std::cout << "foreground channels:" << foregrond_3_CH.channels() << std::endl;
  // std::cout << "foreground size:" << foregrond_3_CH.size() << std::endl;

  // std::cout << "referFrame type:" << type2str(referFrame_3_CH_resized.type()) << std::endl;
  // std::cout << "referFrame channels:" << referFrame_3_CH_resized.channels() << std::endl;
  // std::cout << "referFrame size:" << referFrame_3_CH_resized.size() << std::endl;

  cv::Mat background_3_CH;

  cv::cvtColor(background, background_3_CH, cv::COLOR_GRAY2BGR);

  // std::cout << "background type:" << type2str(background_3_CH.type()) << std::endl;
  // std::cout << "background channels:" << background_3_CH.channels() << std::endl;
  // std::cout << "background size:" << background_3_CH.size() << std::endl;

  cv::Mat out2;

  cv::hconcat(referFrame_3_CH_resized, foregrond_3_CH, out);
  cv::hconcat(out, background_3_CH, out2);

  cv::namedWindow("final", cv::WINDOW_AUTOSIZE);
  // cv::resizeWindow("final", x_size, y_size);
  cv::imshow("final", out2);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

cv::Mat imshow32F(cv::Mat A) {
  CV_Assert(A.type() == CV_32F);

  cv::Mat Ashow(A.rows, A.cols, CV_8U);
  double minval, maxval;
  minMaxIdx(A, &minval, &maxval);
  A.convertTo(Ashow, CV_8U, 255.0 / (maxval - minval), -255.0 * minval / (maxval - minval));
  return Ashow;
}

void colorFlow(cv::Mat flow, std::string figName = "optical flow") {
  // extraxt x and y channels
  cv::Mat xy[2]; // X,Y
  cv::split(flow, xy);

  // calculate angle and magnitude
  cv::Mat magnitude, angle;
  cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

  // translate magnitude to range [0;1]
  double mag_max;
  cv::minMaxLoc(magnitude, 0, &mag_max);
  magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

  // build hsv image
  cv::Mat _hsv[3], hsv;
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magnitude;
  cv::merge(_hsv, 3, hsv);

  // convert to BGR and show
  cv::Mat bgr; // CV_32FC3 matrix
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  cv::imshow(figName, bgr);

  // interpolation

  // imwrite("c://resultOfOF.jpg", bgr);
  // cv::waitKey(0);
}

void drawOptFlowMap(const cv::Mat &flow, cv::Mat &cflowmap, int step, double, const cv::Scalar &color) {
  for (int y = 0; y < cflowmap.rows; y += step)
    for (int x = 0; x < cflowmap.cols; x += step) {
      const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
      line(cflowmap, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
      circle(cflowmap, cv::Point(x, y), 2, color, -1);
    }
}

cv::Mat indexToMask(cv::Mat indexMat, int rows, int cols) {
  cv::Mat maskMat = cv::Mat::zeros(rows, cols, CV_8UC1);
  for (int i = 0; i < indexMat.cols; i++) {
    for (int j = 0; j < indexMat.rows; j++) {
      cv::Vec2i mask_loca = indexMat.at<cv::Vec2i>(j, i);
      if (mask_loca[0] != 0 && mask_loca[1] != 0) {
        maskMat.at<uchar>(cv::Point(mask_loca)) = 255;
      }
    }
  }
  return maskMat;
}

cv::Mat flowHomography(cv::Mat edges, cv::Mat flow, int ransacThre) {
  cv::Mat inlierMask, inlier_edges, inilier_edgeLocations;
  std::vector<cv::Point> edge_Locations1;

  cv::findNonZero(edges, edge_Locations1);

  std::vector<cv::Point> obj_edgeflow;

  for (size_t i = 0; i < edge_Locations1.size(); i++) {
    int src_x = edge_Locations1[i].x;
    int src_y = edge_Locations1[i].y;
    cv::Point2f f = flow.at<cv::Point2f>(src_y, src_x);
    obj_edgeflow.push_back(cv::Point2i(src_x + f.x, src_y + f.y));
  }

  cv::Mat Homography = cv::findHomography(edge_Locations1, obj_edgeflow, cv::RANSAC, ransacThre, inlierMask);

  cv::Mat(edge_Locations1).copyTo(inilier_edgeLocations, inlierMask);

  // convert index matrix to mask matrix
  inlier_edges = indexToMask(inilier_edgeLocations, edges.rows, edges.cols);

  return inlier_edges;
}

cv::Mat sparse_int_dense(cv::Mat im1, cv::Mat im2, cv::Mat im1_edges, cv::Mat sparseFlow) {
  cv::Mat denseFlow;
  std::vector<cv::Point2f> sparseFrom;
  std::vector<cv::Point2f> sparseTo;

  std::vector<cv::Point> edge_Location;
  cv::findNonZero(im1_edges, edge_Location);
  for (size_t i = 0; i < edge_Location.size(); i++) {
    float src_x = edge_Location[i].x;
    float src_y = edge_Location[i].y;
    sparseFrom.push_back(cv::Point2f(src_x, src_y));
    cv::Point2f f = sparseFlow.at<cv::Point2f>(src_y, src_x);
    sparseTo.push_back(cv::Point2f(src_x + f.x, src_y + f.y));
  }

  cv::Ptr<cv::ximgproc::SparseMatchInterpolator> epicInterpolation = cv::ximgproc::createEdgeAwareInterpolator();
  epicInterpolation->interpolate(im1, sparseFrom, im2, sparseTo, denseFlow);
  return denseFlow;
}

void initila_motion_decompose(cv::Mat im1, cv::Mat im2, cv::Mat &back_denseFlow, cv::Mat &fore_denseFlow,
                              int back_ransacThre = 1, int fore_ransacThre = 1) {
  if (im1.channels() != 1)
    cvtColor(im1, im1, cv::COLOR_RGB2GRAY);
  if (im2.channels() != 1)
    cvtColor(im2, im2, cv::COLOR_RGB2GRAY);

  cv::Mat im1_edge, im2_edge;
  cv::Mat flow;
  cv::Mat edgeflow; // extracted edgeflow

  // Mat backH, mask_backH;
  cv::Mat back_edges, rest_edges,
      fore_edges; // edges aligned to the back layer using homography, remaining layer, foreground layers
  cv::Mat back_flow, rest_flow, fore_flow;

  Canny(im1, im1_edge, 10, 100, 3, true);
  Canny(im2, im2_edge, 10, 100, 3, true);

  ///////////////replace edgeflow
  cv::Ptr<cv::DenseOpticalFlow> deepflow = cv::optflow::createOptFlow_DeepFlow();
  deepflow->calc(im1, im2, flow);
  // colorFlow(flow,"optical_flow");
  flow.copyTo(edgeflow, im1_edge);
  // colorFlow(edgeflow,"edge_flow");

  ////////flow=>points using homography-ransac filtering, and then extract flow on the filtered edges
  back_edges = flowHomography(im1_edge, edgeflow, back_ransacThre);
  // imshow("back_edges", back_edges);
  edgeflow.copyTo(back_flow, back_edges);
  // colorFlow(back_flow, "back_flow");
  //////////rest edges and flows
  rest_edges = im1_edge - back_edges;
  // imshow("rest_edges", rest_edges);
  rest_flow = edgeflow - back_flow;
  // colorFlow(rest_flow, "rest_flow");

  ////////////align resting flows to another homograghy
  fore_edges = flowHomography(rest_edges, rest_flow, fore_ransacThre);
  // imshow("fore_edges", fore_edges);
  rest_flow.copyTo(fore_flow, fore_edges);
  // colorFlow(fore_flow, "fore_flow");

  ///////////////////interpolation from sparse edgeFlow to denseFlow/////////////////////
  back_denseFlow = sparse_int_dense(im1, im2, back_edges, back_flow);
  fore_denseFlow = sparse_int_dense(im1, im2, fore_edges, fore_flow);
  // colorFlow(back_denseFlow,"inter_back_denseflow");
  // colorFlow(fore_denseFlow,"inter_fore_denseflow");
}
// flow=flow->cal(im1,im2), so warp im2 to back
cv::Mat imgWarpFlow(cv::Mat im1, cv::Mat flow) {
  cv::Mat flowmap_x(flow.size(), CV_32FC1);
  cv::Mat flowmap_y(flow.size(), CV_32FC1);
  for (int j = 0; j < flowmap_x.rows; j++) {
    for (int i = 0; i < flowmap_x.cols; ++i) {
      cv::Point2f f = flow.at<cv::Point2f>(j, i);
      flowmap_y.at<float>(j, i) = float(j + f.y);
      flowmap_x.at<float>(j, i) = float(i + f.x);
    }
  }
  cv::Mat warpedFrame;
  remap(im1, warpedFrame, flowmap_x, flowmap_y, cv::INTER_CUBIC, cv::BORDER_CONSTANT, 255);
  return warpedFrame;
}

// add flow src_flow + add_flow=>obj_flow
cv::Mat addFlow(cv::Mat src_flow, cv::Mat add_flow) {
  cv::Mat obj_flow = src_flow.clone();
  int src_x, src_y;
  float obj_y, obj_x;

  for (int j = 0; j < src_flow.rows; j++) {
    for (int i = 0; i < src_flow.cols; ++i) {
      cv::Point2f src_f = src_flow.at<cv::Point2f>(j, i);
      src_y = int(j + src_f.y);
      if (src_y >= src_flow.rows) {
        src_y = src_flow.rows - 1;
      }
      src_x = int(i + src_f.x);
      if (src_x >= src_flow.cols) {
        src_x = src_flow.cols - 1;
      }

      cv::Point2f add_f = add_flow.at<cv::Point2f>(src_y, src_x);
      obj_y = float(src_y + add_f.y);
      if (obj_y >= src_flow.rows) {
        obj_y = src_flow.rows - 1;
      }
      obj_x = float(src_x + add_f.x);
      if (obj_x >= src_flow.cols) {
        obj_x = src_flow.cols - 1;
      }
      obj_flow.at<cv::Point2f>(j, i) = cv::Point2f(obj_x - i, obj_y - j);
    }
  }
  return obj_flow;
}

void motion_initiliazor_direct(const std::vector<cv::Mat> &video_input, std::vector<cv::Mat> &back_flowfields,
                               std::vector<cv::Mat> &fore_flowfields, std::vector<cv::Mat> &warpedToReference) {
  int back_ransacThre = 1;
  int fore_ransacThre = 1;

  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    cv::Mat im1, im2; // reference frame, other frame

    // Mat foreH, mask_foreH;
    cv::Mat back_denseFlow, fore_denseFlow;

    if (frame_i != reference_number) {
      // int frame_i=1;
      im1 = video_input[reference_number].clone();
      im2 = video_input[frame_i].clone();

      // decompose motion fields into fore/background
      initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);

      // cout<<back_denseFlow.type()<<endl;
      back_flowfields.push_back(back_denseFlow.clone());
      fore_flowfields.push_back(fore_denseFlow.clone());
      // colorFlow(back_denseFlow,"inter_back_denseflow");
      // colorFlow(fore_denseFlow,"inter_fore_denseflow");
      //
      ////////////warping images to the reference frame///////////////////
      cv::Mat warpedFrame = imgWarpFlow(im2, back_denseFlow);
      warpedToReference.push_back(warpedFrame.clone());
      // imshow("warped image",warpedFrame);
    } else {
      cv::Mat refer_grey = video_input[reference_number].clone();
      warpedToReference.push_back(refer_grey.clone());
      back_flowfields.push_back(cv::Mat::zeros(refer_grey.rows, refer_grey.cols, CV_32FC2));
      fore_flowfields.push_back(cv::Mat::zeros(refer_grey.rows, refer_grey.cols, CV_32FC2));
    }
  }
}

void motion_initiliazor_iterative(const std::vector<cv::Mat> &video_input, std::vector<cv::Mat> &back_flowfields,
                                  std::vector<cv::Mat> &fore_flowfields, std::vector<cv::Mat> &warpedToReference) {
  int back_ransacThre = 1;
  int fore_ransacThre = 1;
  std::vector<cv::Mat> backfields_iterative;
  std::vector<cv::Mat> forefields_iterative;

  cv::Mat im1, im2;
  cv::Mat back_denseFlow, fore_denseFlow, back_iterFLow, fore_iterFlow;
  // flow: 0<-1<-2
  for (size_t frame_i = 0; frame_i < reference_number; frame_i++) {
    im1 = video_input[frame_i + 1].clone();
    im2 = video_input[frame_i].clone();

    initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);
    backfields_iterative.push_back(back_denseFlow.clone());
    forefields_iterative.push_back(fore_denseFlow.clone());
  }

  backfields_iterative.push_back(cv::Mat::zeros(im2.rows, im2.cols, CV_32FC2));
  forefields_iterative.push_back(cv::Mat::zeros(im2.rows, im2.cols, CV_32FC2));
  // flow: 2->3->4
  for (size_t frame_i = reference_number; frame_i < (frameNumber - 1); frame_i++) {
    im1 = video_input[frame_i].clone();
    im2 = video_input[frame_i + 1].clone();

    initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);
    backfields_iterative.push_back(back_denseFlow.clone());
    forefields_iterative.push_back(fore_denseFlow.clone());
    //        colorFlow(back_denseFlow,"inter_back_denseflow");
    //        colorFlow(fore_denseFlow,"inter_fore_denseflow");
  }
  //
  ////////////warping images to the reference frame///////////////////
  for (size_t frame_i = 0; frame_i < frameNumber; frame_i++) {
    im2 = video_input[frame_i].clone();
    back_denseFlow =
        cv::Mat::zeros(im2.rows, im2.cols, CV_32FC2); // accumulate flow to the reference frame by iterative warping
    fore_denseFlow = cv::Mat::zeros(im2.rows, im2.cols, CV_32FC2);

    if (frame_i == reference_number) {
      warpedToReference.push_back(im2.clone());
      back_flowfields.push_back(back_denseFlow.clone());
      fore_flowfields.push_back(fore_denseFlow.clone());
    } else {
      for (int ii = 0; ii < abs(int(reference_number) - int(frame_i)); ii++) {
        int itera_ii = (int(reference_number) - int(frame_i)) > 0 ? (frame_i + ii) : (frame_i - ii);
        back_iterFLow = backfields_iterative[itera_ii].clone();
        fore_iterFlow = forefields_iterative[itera_ii].clone();
        ////sometimes foreground flow is better for reflections
        // Mat warped=imgWarpFlow(im2, fore_iterFlow);
        cv::Mat warped = imgWarpFlow(im2, back_iterFLow);
        im2 = warped.clone();
        std::cout << frame_i << reference_number << itera_ii << std::endl;

        if (ii > 0) {
          back_denseFlow = addFlow(back_denseFlow, back_iterFLow);
          fore_denseFlow = addFlow(fore_denseFlow, fore_iterFlow);
        } else {
          back_denseFlow = back_iterFLow.clone();
          fore_iterFlow = fore_iterFlow.clone();
        }
      }
      warpedToReference.push_back(im2.clone());
      back_flowfields.push_back(back_denseFlow.clone());
      fore_flowfields.push_back(fore_denseFlow.clone());
      colorFlow(back_denseFlow, "inter_back_denseflow");
      colorFlow(fore_denseFlow, "inter_fore_denseflow");
    }
  }
}

cv::Mat Laplac(const cv::Mat &input) {
  CV_Assert(input.type() == CV_8U || input.type() == CV_32F);

  cv::Mat _input;
  if (input.type() == CV_8U)
    input.convertTo(_input, CV_32F);
  else
    _input = input.clone();

  int width = input.cols;
  int height = input.rows;

  _input = _input.reshape(0, 1);
  cv::Size s = _input.size();
  cv::Mat temp = cv::Mat::zeros(s, CV_32FC1);
  cv::Mat _output = cv::Mat::zeros(s, CV_32FC1);
  cv::Mat output = cv::Mat::zeros(height, width, CV_32FC1);

  // horizontal filtering
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width - 1; j++) {
      int offset = i * width + j;
      temp.at<float>(offset) = _input.at<float>(offset + 1) - _input.at<float>(offset);
    }

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int offset = i * width + j;
      if (j < width - 1)
        _output.at<float>(offset) -= temp.at<float>(offset);
      if (j > 0)
        _output.at<float>(offset) += temp.at<float>(offset - 1);
    }

  temp.release();

  temp = cv::Mat::zeros(s, CV_32FC1);

  // vertical filtering
  for (int i = 0; i < height - 1; i++)
    for (int j = 0; j < width; j++) {
      int offset = i * width + j;
      temp.at<float>(offset) = _input.at<float>(offset + width) - _input.at<float>(offset);
    }
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int offset = i * width + j;
      if (i < height - 1)
        _output.at<float>(offset) -= temp.at<float>(offset);
      if (i > 0)
        _output.at<float>(offset) += temp.at<float>(offset - width);
    }

  output = _output.reshape(0, input.rows);
  // output.convertTo(output, input.type());
  return output;
}

void motDecomIrlsWeight(const std::vector<cv::Mat> &input_sequence, const cv::Mat &backgd_comp,
                        const cv::Mat &obstruc_comp, cv::Mat &alpha_map, const std::vector<cv::Mat> &back_flowfields,
                        const std::vector<cv::Mat> &fore_flowfields, std::vector<float> &omega_1,
                        std::vector<float> &omega_2, std::vector<float> &omega_3) {

  int width = backgd_comp.cols;
  int height = backgd_comp.rows;
  int npixels = width * height;

  CV_Assert(omega_1.size() == 0 || omega_1.size() == input_sequence.size() * backgd_comp.total());
  CV_Assert(omega_2.size() == 0 || omega_2.size() == backgd_comp.total());
  CV_Assert(omega_3.size() == 0 || omega_3.size() == backgd_comp.total());

  float varepsilon = pow(0.001, 2);
  int deriv_ddepth = CV_32F;

  cv::Mat backgd_dx;
  cv::Mat backgd_dy;
  cv::Mat obstruc_dx;
  cv::Mat obstruc_dy;

  // compute gradients of current background and occlusion components
  cv::Sobel(backgd_comp, backgd_dx, deriv_ddepth, 1, 0);
  cv::Sobel(backgd_comp, backgd_dy, deriv_ddepth, 0, 1);
  cv::Sobel(obstruc_comp, obstruc_dx, deriv_ddepth, 1, 0);
  cv::Sobel(obstruc_comp, obstruc_dy, deriv_ddepth, 0, 1);

  // if weights have not been initialized
  if (omega_1.size() == 0 && omega_2.size() == 0 && omega_3.size() == 0) {

    // compute derivative denominators (weights)
    for (size_t tt = 0; tt < input_sequence.size(); tt++) {
      cv::Mat img = input_sequence[tt];
      cv::Mat back_flow = back_flowfields[tt];
      cv::Mat obstruc_flow = fore_flowfields[tt];
      cv::Mat temp = img - imgWarpFlow(obstruc_comp, obstruc_flow) -
                     imgWarpFlow(alpha_map, obstruc_flow).mul(imgWarpFlow(backgd_comp, back_flow));
      for (int i = 0; i < npixels; i++) {
        omega_1.push_back(1 / sqrt(temp.at<float>(i) * temp.at<float>(i) + varepsilon));
      }
    }
    for (int i = 0; i < npixels; i++) {
      omega_2.push_back(1 / sqrt(backgd_dx.at<float>(i) * backgd_dx.at<float>(i) +
                                 backgd_dy.at<float>(i) * backgd_dy.at<float>(i) + varepsilon));
      omega_3.push_back(1 / sqrt(obstruc_dx.at<float>(i) * obstruc_dx.at<float>(i) +
                                 obstruc_dy.at<float>(i) * obstruc_dy.at<float>(i) + varepsilon));
    }
  }

  // if weights have been calculated
  else if (omega_1.size() == input_sequence.size() * backgd_comp.total() && omega_2.size() == backgd_comp.total() &&
           omega_3.size() == backgd_comp.total()) {

    // compute derivative denominators (weights)
    for (size_t tt = 0; tt < input_sequence.size(); tt++) {
      cv::Mat img = input_sequence[tt];
      cv::Mat back_flow = back_flowfields[tt];
      cv::Mat obstruc_flow = fore_flowfields[tt];
      cv::Mat temp = img - imgWarpFlow(obstruc_comp, obstruc_flow) -
                     imgWarpFlow(alpha_map, obstruc_flow).mul(imgWarpFlow(backgd_comp, back_flow));

      for (int i = 0; i < npixels; i++) {
        int offset = i + tt * npixels;
        omega_1[offset] = 1 / sqrt(temp.at<float>(i) * temp.at<float>(i) + varepsilon);
      }
    }
    for (int i = 0; i < npixels; i++) {
      omega_2[i] = 1 / sqrt(backgd_dx.at<float>(i) * backgd_dx.at<float>(i) +
                            backgd_dy.at<float>(i) * backgd_dy.at<float>(i) + varepsilon);
      omega_3[i] = 1 / sqrt(obstruc_dx.at<float>(i) * obstruc_dx.at<float>(i) +
                            obstruc_dy.at<float>(i) * obstruc_dy.at<float>(i) + varepsilon);
    }
  }
}
