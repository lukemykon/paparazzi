#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// Sorting points at Y-value 
bool sortByY(const Point& a, const Point& b) { return a.y < b.y; }
// Sorting points at X-value
bool sortByX(const Point& a, const Point& b) { return a.x < b.x; }

// Beginning might need some adjustments based on dataset 
int main() {
    string folder_path = "/home/jort/Edge_detection/AE4317_2019_datasets/sim_poles_panels/20190121-161422/";
    vector<String> filenames;
    glob(folder_path + "*.jpg", filenames); // Collects all .jpg files in the folder

    double alpha = 0.8;
    Vec4f smoothed_left(0,0,0,0), smoothed_right(0,0,0,0);
    bool init_left = false, init_right = false;

    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i]);
        if (img.empty()) continue;

        Mat blurred, hsv, mask;
        GaussianBlur(img, blurred, Size(5, 5), 0);
        cvtColor(blurred, hsv, COLOR_BGR2HSV);

        // 1. Green Field Masking
        Scalar lower_green(30, 100, 80), upper_green(55, 255, 180);
        inRange(hsv, lower_green, upper_green, mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat clean_mask = Mat::zeros(img.size(), CV_8UC1);
        int top_y = 0;

        if (!contours.empty()) {
            double max_area = 0;
            int max_idx = -1;
            for (int j = 0; j < contours.size(); j++) {
                double area = contourArea(contours[j]);
                if (area > max_area) { max_area = area; max_idx = j; }
            }

            if (max_idx != -1 && max_area > (img.rows * img.cols * 0.05)) {
                vector<Point> hull;
                convexHull(contours[max_idx], hull);
                fillConvexPoly(clean_mask, hull, Scalar(255));
                
                Rect br = boundingRect(hull);
                top_y = br.y;
                clean_mask(Rect(0, 0, img.cols, max(0, top_y - 10))).setTo(0);
            }
        }

        // 2. Line Detection 
        Mat edges;
        Canny(clean_mask, edges, 50, 150);
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI/180, 40, 60, 100);

        Vec4f curr_left, curr_right;
        bool found_left = false, found_right = false;

        for (auto& l : lines) {
            float dx = l[2] - l[0];
            float dy = l[3] - l[1];
            if (dx == 0) continue;
            float slope = dy / dx;

            if (slope > 0.2) { curr_left = l; found_left = true; }
            else if (slope < -0.2) { curr_right = l; found_right = true; }
        }

        // Smoothing (EMA)
        if (found_left) {
            if (!init_left) { smoothed_left = curr_left; init_left = true; }
            else smoothed_left = alpha * curr_left + (1.0 - alpha) * smoothed_left;
        }
        if (found_right) {
            if (!init_right) { smoothed_right = curr_right; init_right = true; }
            else smoothed_right = alpha * curr_right + (1.0 - alpha) * smoothed_right;
        }

        Mat output_img = img.clone();
        if (init_left) line(output_img, Point(smoothed_left[0], smoothed_left[1]), Point(smoothed_left[2], smoothed_left[3]), Scalar(0, 255, 255), 4);
        if (init_right) line(output_img, Point(smoothed_right[0], smoothed_right[1]), Point(smoothed_right[2], smoothed_right[3]), Scalar(255, 0, 255), 4);

        // 3. Panel Detection
        Mat gray, dark, dark_on_mat;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        threshold(gray, dark, 75, 255, THRESH_BINARY_INV);
        bitwise_and(dark, clean_mask, dark_on_mat);

        Mat kernel_p = getStructuringElement(MORPH_RECT, Size(9, 9));
        morphologyEx(dark_on_mat, dark_on_mat, MORPH_CLOSE, kernel_p);

        vector<vector<Point>> p_contours;
        findContours(dark_on_mat, p_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto& pcnt : p_contours) {
            if (contourArea(pcnt) > 1000) {
                vector<Point> approx;
                approxPolyDP(pcnt, approx, 0.03 * arcLength(pcnt, true), true);
                if (approx.size() >= 5 && approx.size() <= 6) {
                    RotatedRect rr = minAreaRect(pcnt);
                    Point2f vtx[4];
                    rr.points(vtx);
                    for (int j = 0; j < 4; j++) line(output_img, vtx[j], vtx[(j+1)%4], Scalar(255, 0, 0), 3);
                    putText(output_img, "Panel", rr.center, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
                }
            }
        }

        // 4. Gate Detection (Orange)
        Mat gate_mask;
        inRange(hsv, Scalar(5, 100, 100), Scalar(35, 255, 255), gate_mask);
        morphologyEx(gate_mask, gate_mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));

        vector<vector<Point>> g_contours;
        findContours(gate_mask, g_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto& gcnt : g_contours) {
            Rect br = boundingRect(gcnt);
            float ar = (float)br.width / br.height;
            if (contourArea(gcnt) > 800 && ar > 0.6 && ar < 1.6) {
                vector<Point> approx;
                approxPolyDP(gcnt, approx, 0.04 * arcLength(gcnt, true), true);
                if (approx.size() >= 4 && approx.size() <= 8) {
                    // Angle calculation
                    vector<Point> pts = approx;
                    sort(pts.begin(), pts.end(), sortByY);
                    vector<Point> top_pts = {pts[0], pts[1]};
                    sort(top_pts.begin(), top_pts.end(), sortByX);

                    float dx = top_pts[1].x - top_pts[0].x;
                    float dy = top_pts[1].y - top_pts[0].y;
                    double angle = (dx != 0) ? atan2(dy, dx) * 180.0 / CV_PI : 0;

                    polylines(output_img, approx, true, Scalar(255, 255, 0), 3);
                    string status = (abs(angle) < 2) ? "Straight" : "Angle: " + to_string(angle).substr(0,4) + " deg";
                    putText(output_img, status, Point(br.x, br.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, (abs(angle) < 2 ? Scalar(0,255,0) : Scalar(0,0,255)), 2);
                }
            }
        }

        imshow("Result", output_img);
        if (waitKey(30) == 'q') break;
    }
    return 0;
}
