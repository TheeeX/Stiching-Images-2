#include<opencv\cv.h>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/** @function main */
int main(){

	/*-- Load the images --*/
	Mat image1 = imread("C:\\panL.jpg");
	Mat image2 = imread("C:\\panR.jpg");

	if (!image1.data || !image2.data)
	{
		cout << " --(!) Error reading images " << endl; return -1;
	}

	imshow("first image", image2);
	imshow("second image", image1);

	/*-- Detecting the keypoints using SURF Detector --*/
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);

	vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(image1, keypoints_1);
	detector.detect(image2, keypoints_2);

	/*-- Calculating descriptors (feature vectors) --*/
	SurfDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;

	extractor.compute(image1, keypoints_1, descriptors_1);
	extractor.compute(image2, keypoints_2, descriptors_2);

	/*-- Step 3: Matching descriptor vectors using FLANN matcher --*/
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	//-- Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	cout << "-- Max dist :" << max_dist << endl;
	cout << "-- Min dist :" << min_dist << endl;

	/*-- Drawing matches  whose distance is less than 2*min_dist,
	 *-- or a small arbitary value ( 0.02 ) in the event that min_dist is verysmall)
	 */
	vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}
	/*-- Draw only good matches --*/
	Mat img_matches;
	drawMatches(image1, keypoints_1, image2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	/*-- Show detected matches --*/
	imshow("Good Matches", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		cout << "-- Good Match [i] Keypoint 1: " << good_matches[i].queryIdx << " -- Keypoint 2:" << good_matches[i].trainIdx << endl;
	}

	waitKey(0);
	return 0;
}