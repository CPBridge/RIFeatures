// This file provides a minimal example on using the RIFeatures library to
// extract roatation invariant features from the frames of a video sequence

// Standard library
#include <iostream> /* cout, endl, cerr */
#include <cmath> /* min */

// OpencCV modules that we will need
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// The rotation invariant features header file
#include <RIFeatures/RIFeatExtractor.hpp>

// Main routine
int main( int argc, char** argv )
{
	// Make sure the user has provided a single video file name as input
	if(argc != 2)
	{
		std::cerr << "ERROR: A single video file name must be passed as a command line argument!" << std::endl;
		return EXIT_FAILURE;
	}

	// Open the video file
	cv::VideoCapture vid_obj;
	vid_obj.open(argv[1]);
	if ( !vid_obj.isOpened() )
	{
		std::cerr  << "ERROR: Could not open video file " << argv[1] << std::endl;
		return EXIT_FAILURE;
	}

	// Get the dimensions of the video
	const int xsize = vid_obj.get(cv::CAP_PROP_FRAME_WIDTH);
	const int ysize = vid_obj.get(cv::CAP_PROP_FRAME_HEIGHT);

	// Parameters of the feature extractor object setup:

	// The size of the input frames from the video
	const cv::Size image_size(xsize,ysize);

	// The radius of the detection window.
	// Let's arbitrarily choose 1/10 of the smallest of the two image dimensions
	const float detection_radius = std::min(xsize,ysize)/10.0;

	// The J,K and M parameters together define the set of raw features
	// that can be produced by the extraction object. Let's choose J=4, K=4, M=4
	const int J = 4, K = 4, M = 4;

	// Create a feature extraction object using these parameters.
	// There are a number of other parameters that we can control here, but let's
	// just accept the default values.
	RIFeatures::RIFeatExtractor feature_extractor(image_size, detection_radius, J, K, M);

	// Now we can find out how many raw and derived features are available
	// from this feature extractor
	std::cout << "There are " << feature_extractor.getNumRawFeats() << " raw features and " << feature_extractor.getNumDerivedFeats() << " derived features available from this feature extraction object." << std::endl;

	// We need a list of image locations at which we want to extract some features
	// These need to be specified as cv::Points that define the centre of the
	// detection window. We need to make sure that these points are at least
	// a distance of getMaxSpatBasisHalfsize() from the edge of the image.
	// To make things easy we'll just do a naive loop and take the first 20 points
	// that satisfy these conditions.
	std::vector<cv::Point> image_locations;
	int x = 0, y = 0;
	const int border_width = feature_extractor.getMaxSpatBasisHalfsize();
	while(image_locations.size() < 20)
	{
		if( (x >= border_width) &&
		    (x < xsize - border_width) &&
		    (y >= border_width) &&
			(y < ysize - border_width)
		  )
		{
			image_locations.emplace_back(cv::Point(x,y));
		}
		++x;
		if(x >= xsize)
		{
			x = 0;
			++y;
		}
	}

	// We also need to choose the index of a derived feature to extract each
	// time. Let's arbitrarily choose one about halfway through the list of
	// possible values
	const int query_feature_index = feature_extractor.getNumDerivedFeats()/2;

	// Start a loop through the video frames
	cv::Mat frame;
	int f = 0; // frame counter
	while(true)
	{
		// Grab next frame
		vid_obj >> frame;
		if(frame.empty())
		{
			// We have reached the end of the video
			break;
		}

		// If the frame is RGB, convert to greyscale
		if(frame.channels() > 1)
		{
			cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
		}

		// Calculate the gradient image using OpenCV's Sobel filter
		// Note that in principle any 1 or 2-channel representation derived
		// from the input frame could be used instead (e.g. intensity or motion)
		cv::Mat_<float> gradient_planes[2];
		static constexpr int C_SOBEL_KERNEL_SIZE = 5;
		cv::Sobel(frame,gradient_planes[0],CV_32F,1,0,C_SOBEL_KERNEL_SIZE,1.0);
		cv::Sobel(frame,gradient_planes[1],CV_32F,0,1,C_SOBEL_KERNEL_SIZE,1.0);

		// The feature extractor expects a vector field in a polar representation
		// consisting of a magnitude image and an orientation image, in radians
		// and defined according to standard mathematical convention (i.e.
		// anticlockwise from the increasing x-axis)
		cv::Mat_<float> magnitude_image, orientation_image;
		gradient_planes[1] *= -1.0; // need to swap the sign of y componenent to match convention
		cv::cartToPolar(gradient_planes[0],gradient_planes[1],magnitude_image,orientation_image);

		// Place into the feature extractor
		feature_extractor.setVectorInputImage(magnitude_image, orientation_image);

		// Now the image has been placed in the extractor object we can query
		// it to get the values of the derived rotation invariant features.
		std::vector<float> feature_values(image_locations.size());
		feature_extractor.getDerivedFeature(image_locations.cbegin(),image_locations.cend(),query_feature_index,feature_values.begin());

		// Output derived feature values
		std::cout << "Frame " << f << ": ";
		for(float val : feature_values)
			std::cout << val << " ";
		std::cout << std::endl;

		++f; // frame counter
	}

}
