// Written by Christopher Bridge for the RIFeatures library. Licensed under the
// GNU general public license version 3. 

#ifndef RIFEATEXTRACTOR_H
#define RIFEATEXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> /* mostly for debug purposes */
#include <vector>
#include <complex> /* complex numbers */
#include <omp.h> /* OpenMP for multi-threading */

namespace RIFeatures
{

/*! \brief This is the main class that encapsulates the rotation invariant feature
* extraction process
*
* Once set up, images may be 'placed into' the extractor and then queried
* to get various features. More details will be found in my forthcoming DPhil thesis.
*/
class RIFeatExtractor
{
	public:
		// Data Types

		// Calculation methodologies
		/*! Enumeration representing the methods that the class can use
		* to perform the calculations.
		*/
		enum calculationMethod_enum : unsigned char
		{
			cmSpatial = 0, //!< Use spatial domain calculations (convolutions)
			cmFrequency, //!< Use frequency-domain calculations via the fast Fourier transform
			cmAuto //!< Automatically determine the best method to use based on the number of image locations queried (not guaranteed to be fastest!)
		};

		// Coupling methodologies
		/*! Enumeration representing the methods the class can use to perform
		* coupling calculations between pairs of raw features
		*/
		enum couplingMethod_enum : unsigned char
		{
			comElementwise = 0, //!< Perform the coupling coupling on each pixel at a time
			comVectorised, //!< Perform coupling on entire images at a time
			comAuto //!< Automatically determine which method to use based on the number of locations queried (not guaranteed to be fastest!)
		};

		// Feature sets
		/*! Enumeration representing the feature set used by the extractor
		*/
		enum featureSet_enum : unsigned char
		{
			fsBasic = 0, //!< Basic feature set, with no feature coupling
			fsSimpleCouple, //!< Feature set with coupling between features of the same rotation order
			fsExtraCouple //!< Feature set with coupling between all possible pairs of features
		};

		// Types of basis function
		/*! Enumeration representing the type of basis functions used by the extracor */
		enum basisType_enum : unsigned char
		{
			btSoftHist = 0, //!< Basis functions with a soft histogram radial profile
			btZernike //!< Basis functions with based on Zernike moments (NB this has not been well maintained and may be incompatible with certain other features)
		};

		// Methods
		RIFeatExtractor();
		RIFeatExtractor(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method = cmAuto, const bool use_spatial_memoisation = false, const couplingMethod_enum couple_method = comAuto, const featureSet_enum feature_set = fsBasic, const int max_derived_rotation_order = -1, const basisType_enum basis_form = btSoftHist); // constructor
		~RIFeatExtractor(void); // destructor
		void initialise(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method = cmAuto, const bool use_spatial_memoisation = false, const couplingMethod_enum couple_method = comAuto, const featureSet_enum feature_set = fsBasic, const int max_derived_rotation_order = -1, const basisType_enum basis_form = btSoftHist);
		void createRawFeats();
		int getNumRawFeats() const;
		int getNumDerivedFeats() const;
		int getMaxSpatBasisHalfsize(const int Jmax = -1) const;
		void getFeatsUsingLowJ(const int Jmax, std::vector<int>& output_feature_list) const;
		void getFeatsWithGivenR(const int r, std::vector<int> &raw_feat_ind, const bool include_negatives = false, const int Jmax = -1) const;
		template<typename TInputIterator,typename TOutputIterator>
		void getDerivedFeature(TInputIterator first_point, const TInputIterator last_point, const int derived_feat_num, TOutputIterator out);
		template<typename TInputIterator,typename TOutputIterator>
		void getRawFeatureArg(TInputIterator first_point, const TInputIterator last_point, const int raw_feature_index, TOutputIterator cos_dest, TOutputIterator sin_dest, const bool flip_negative_rotation_orders = false);
		cv::Mat getSpatialBasisCopy(const int basis_index) const;
		cv::Mat getFrequencyBasisCopy(const int basis_index) const;
		int getNumBases() const;
		bool getBasisInfo(const int basis_index, int& j, int& k) const;
		void setScalarInputImage(const cv::Mat &in);
		void setVectorInputImage(const cv::Mat &in_magnitude, const cv::Mat &in_orientation);
		std::complex<float> singleWindowFeature(const int raw_feature_num, const cv::Point p);
		static bool stringToCalcMethod(const std::string& method_string, calculationMethod_enum& method_enum);
		static bool stringToCoupleMethod(const std::string& method_string, couplingMethod_enum& method_enum);
		static bool stringBasisType(const std::string& basis_type_string, basisType_enum& basis_type_enum);
		static bool stringToFeatureSet(const std::string& feature_set_string, featureSet_enum& feature_set_enum);

	private:

		enum featureType_enum : unsigned char
		{
			ftMagnitude = 0,
			ftReal,
			ftImaginary,
		};

		// Methods
		void createDerivedFeatureList(const featureSet_enum feature_set);
		void findSpatSupport();
		void spatBasisSoftHist(const int u, const int j, const int k);
		void freqBasisSoftHist();
		void spatBasisZernike(const int u, const int j, const int k);
		void freqBasisZernike();
		int speedTestForThresh();
		int speedTestForCoupling();
		void expandFourierImages();
		void refreshImage();
		void rawFeatureFrequencyCalculation(const int f);
		void rawFeatureFrequencyCalculation(const int u, const int m);
		void rawFeatureFrequencyCalculation(const int f, const int u, const int m);
		template<typename TInputIterator,typename TOutputIterator>
		void getDerivedFromSingleRawFeature(const int f, const featureType_enum type, TInputIterator first_point, const TInputIterator last_point, TOutputIterator dest);
		void fullImageCouple(cv::Mat_<cv::Vec2f>& coupled_image, const int f1, const int f2);
		bool checkRawFeatureValidity(const int f, const bool calculate_if_invalid);
		int ensureMagnitudeImageValidity(const int f);
		bool checkCoupledImageValidity(const int f1, const int f2, const bool calculate_if_invalid, int& index);
		static void raiseComplexImageToPower(const cv::Mat_<cv::Vec2f>& in, cv::Mat_<cv::Vec2f>& out, const int power);
		static int factorial(const int n);
		static float struveLambda0(const float x);
		static float struvePhi(const float x);
		static float coneHankel(const float rho, const float a, const int k);

		// Data
		std::vector<cv::Mat_<cv::Vec2f>> U_freq; // the frequency-domain basis function images
		std::vector<cv::Mat_<cv::Vec2f>> U_spat; // the time-domain basis function images
		std::vector<cv::Mat_<cv::Vec2f>> raw_feats_unpadded;
		std::vector<cv::Mat_<cv::Vec2f>> raw_feat_images;
		cv::Mat_<float> I; // the (magnitude) input image
		cv::Mat_<float> I_ori; // the orientation input image
		cv::Mat_<cv::Vec2f> I_2chan; // a double channeled input image for quick image domain calculations
		std::vector<cv::Mat_<cv::Vec2f>> I_fou; // the expanded Fourier coefficient images
		cv::Mat_<float> R;
		cv::Mat_<float> theta;
		std::vector<cv::Mat_<cv::Vec2f>> FFT_im;
		cv::Mat_<float> planes[2];
		std::vector<char> raw_features_valid, magnitude_image_valid, coupled_image_valid; // do not change this to vector<bool> or all the threading breaks!!!
		std::vector<std::vector<int>> coupled_image_index_lookup;
		float basis_radius, r_space, area_normaliser;
		std::vector<std::vector<std::complex<float>>> spat_memoiser;
		std::vector<std::vector<char>> spat_memoiser_valid;
		std::vector<cv::Mat_<float>> raw_feature_magnitude_images;
		std::vector<cv::Mat_<cv::Vec2f>> coupled_images;
		int nj, nk, nm;
		int ysize, xsize, pad_ysize, pad_xsize, num_pixels;
		int num_bases;
		basisType_enum basis_type;
		int max_r;
		int spat_basis_size, spat_basis_half_size;
		int use_spatial_threshold, use_pixelwise_coupling_threshold;
		int num_raw_features, num_derived_features, num_magnitude_features, num_coupled_images;
		std::vector<int> basis_j_list, basis_k_list;
		std::vector<int> raw_feat_j_list, raw_feat_k_list, raw_feat_m_list, raw_feat_r_list, raw_feat_basis_list;
		std::vector<int> derived_feature_primary_list, derived_feature_secondary_list;
		std::vector<featureType_enum> derived_feature_type_list;
		std::vector<int> index_lookup_table;
		std::vector<int> raw_feat_usage_last_frame, raw_feat_usage_this_frame;
		std::vector<int> raw_feat_to_magnitude_index;
		bool use_memoiser, use_frequency, auto_method, always_use_frequency;
		bool use_vectorised_coupling, always_use_vectorised_coupling;
		float dummy_result_float;
		std::vector<omp_lock_t> raw_feat_frequency_creation_thread_lock, magnitude_images_creation_locks, coupled_images_creation_locks;

		// Constants
		static constexpr int C_SPEED_TEST_RUNS = 10;
		static constexpr int C_COUPLING_SPEED_TEST_NUM = 10000;

		// Feature types
		static constexpr int C_SECOND_FEATURE_NONE = -1;


};

} // end of namspace

// Include the template implementation file
#include <RIFeatures/RIFeatExtractor.tpp>

#endif
