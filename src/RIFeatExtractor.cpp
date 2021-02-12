// Written by Christopher Bridge for the RIFeatures library. Licensed under the
// GNU general public license version 3.

#include <cmath>
#include <limits>
#include <cassert>
#include <boost/math/special_functions/bessel.hpp>
#include <RIFeatures/struve.hpp>
#include <RIFeatures/RIFeatExtractor.hpp>

namespace RIFeatures
{

/*! \brief Default constructor
*
* Note that an object constructed this way will have no functionality until initialised
*/
RIFeatExtractor::RIFeatExtractor()
{

}

/*! \brief Constructor with initialisation
*
* Constructs a fully functional object. Defines various parameters of the object. This may be slow due to the need to precalculate the basis functions.
* \param image_size Size that the input images will be
* \param basis_diameter Diameter (in pixels) of the circular detection window
* \param num_radii Number of radial profiles in the set of basis functions (parameter J).
* Note that negative values are not permitted and will be silently treated as 0.
* \param num_rot_orders Maximum rotation order in the set of basis functions (parameter K).
* Note that currently values above 6 are not implemented and will be treated as 6.
* \param num_fourier_coefs Number of Fourier coefficients in the orientation histogram expansion (parameter M).
* Note that negative values are not permitted and will be silently treated as 0.
* \param method Calculation method the object will use to calulate the raw features
* \param use_spatial_memoisation If true, use a memoiser to store values calculated
* using the spatial method. This will have no effect if method is set to \c cmFrequency.
* \param couple_method Method to use for coupling calculations between raw features
* \param feature_set Type of derived features that can be extracted. Governs what type of coupling is available
* \param max_derived_rotation_order Maximum rotation order of derived features
* \param basis_form The form of the basis functions.
*/
RIFeatExtractor::RIFeatExtractor(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method, const bool use_spatial_memoisation, const couplingMethod_enum couple_method, const featureSet_enum feature_set, const int max_derived_rotation_order, const basisType_enum basis_form)
{
	// Simply call the initialisation function
	initialise(image_size,basis_diameter,num_radii,num_rot_orders,num_fourier_coefs,method,use_spatial_memoisation,couple_method, feature_set, max_derived_rotation_order, basis_form);
}

// Initialise - creates all the necessary arrays to make the object ready for use
/*! \brief Initialises or re-initialises an object.
*
* Defines various parameters of the object. This may be slow due to the need to precalculate the basis functions.
* \param image_size Size that the input images will be
* \param basis_diameter Diameter (in pixels) of the circular detection window
* \param num_radii Number of radial profiles in the set of basis functions (parameter J).
* Note that negative values are not permitted and will be silently treated as 0.
* \param num_rot_orders Maximum rotation order in the set of basis functions (parameter K).
* Note that currently values above 6 are not implemented and will be treated as 6.
* \param num_fourier_coefs Number of Fourier coefficients in the orientation histogram expansion (parameter M).
* Note that negative values are not permitted and will be silently treated as 0.
* \param method Calculation method the object will use to calulate the raw features
* \param use_spatial_memoisation If true, use a memoiser to store values calculated
* using the spatial method. This will have no effect if method is set to \c cmFrequency.
* \param couple_method Method to use for coupling calculations between raw features
* \param feature_set Type of derived features that can be extracted. Governs what type of coupling is available
* \param max_derived_rotation_order Maximum rotation order of derived features
* \param basis_form The form of the basis functions.
*/
void RIFeatExtractor::initialise(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method, const bool use_spatial_memoisation, const couplingMethod_enum couple_method, const featureSet_enum feature_set, const int max_derived_rotation_order, const basisType_enum basis_form)
{
	// Copy across input parameters to member data
	xsize = image_size.width;
	ysize = image_size.height;
	basis_radius = basis_diameter/2.0;
	nj = (num_radii < 0) ? 0 : num_radii;
	nk = (num_rot_orders > 6) ? 6 : num_rot_orders;
	nm = (num_fourier_coefs < 0) ? 0 : num_fourier_coefs;
	max_r = max_derived_rotation_order;
	basis_type = basis_form;

	// Set up parameters for Fourier transforming the incoming images
	pad_xsize = cv::getOptimalDFTSize(xsize);
	pad_ysize = cv::getOptimalDFTSize(ysize);
	planes[1] = cv::Mat::zeros(pad_ysize,pad_xsize,CV_32F);

	// Create the basis functions
	switch(basis_type)
	{
		default:
		case btSoftHist:
			num_bases = (nm == 0) ? nj*(nk+1) : nj*(2*nk+1);
			break;
		case btZernike:
			if (nm == 0)
				num_bases = (nj%2 == 0) ? nj*(2+nj)/4 : (nj+1)*(nj+1)/4;
			else
				num_bases = nj*(nj+1)/2;
			break;
	} // end switch

	use_frequency = ((method == cmAuto) || (method == cmFrequency));
	always_use_frequency = (method == cmFrequency);
	auto_method = (method == cmAuto);
	use_vectorised_coupling = ((feature_set == fsSimpleCouple) || (feature_set == fsExtraCouple)) && use_frequency && ((couple_method == comAuto) || (couple_method == comVectorised));
	always_use_vectorised_coupling = use_frequency && (couple_method == comVectorised);
	use_memoiser = use_spatial_memoisation && (!always_use_frequency);
	if(use_frequency)
		U_freq.resize(num_bases);
	U_spat.resize(num_bases);
	basis_j_list.resize(num_bases);
	basis_k_list.resize(num_bases);
	r_space = basis_radius/nj;
	area_normaliser = basis_radius*basis_radius;
	findSpatSupport();

	switch(basis_type)
	{
		case btSoftHist:
			if(use_frequency)
				freqBasisSoftHist();

			for(int j = 0, u = 0; j < nj; ++j)
			{
				for( int k = (nm==0) ? 0 : -nk; k <= nk; ++k)
				{
					spatBasisSoftHist(u,j,k);
					basis_j_list[u] = j;
					basis_k_list[u] = k;
					++u;
				} // k loop
			} // j loop
			break;

		case btZernike:
			if(use_frequency)
				freqBasisZernike();

			for(int j = 0, u = 0; j < nj; ++j)
			{
				for( int k = (nm==0) ? j%2 : -j; k <= j; k+=2)
				{
					spatBasisZernike(u,j,k);
					basis_j_list[u] = j;
					basis_k_list[u] = k;
					++u;
				} // k loop
			} // j loop
			break;

	}

	// Create array of Fourier transformed images
	FFT_im.resize(nm+1);

	// Loop to find the parameters of each raw feature
	// image
	raw_feat_j_list.clear();
	raw_feat_k_list.clear();
	raw_feat_m_list.clear();
	raw_feat_r_list.clear();
	raw_feat_basis_list.clear();
	index_lookup_table.resize((nm+1)*num_bases);
	for(int m = 0, f = 0; m <= nm; ++m)
	{
		for(int u = 0; u < num_bases; ++u)
		{
			const int k = basis_k_list[u];
			const int j = basis_j_list[u];
			if((m == 0) && (k < 0)) continue;
			if((max_r >= 0) && std::abs(m-k) > max_r) continue;
			index_lookup_table[m*num_bases+u] = raw_feat_j_list.size();
			raw_feat_j_list.emplace_back(j);
			raw_feat_k_list.emplace_back(k);
			raw_feat_m_list.emplace_back(m);
			raw_feat_r_list.emplace_back(k-m);
			raw_feat_basis_list.emplace_back(u);
			++f;
		}
	}
	num_raw_features = raw_feat_j_list.size();

	if(use_frequency)
	{
		// Allocate memory for the results
		raw_feat_images.resize(num_raw_features);
		raw_feats_unpadded.resize(num_raw_features);

		// Flags that are used to record whether the raw feature images are
		// valid
		// Destroy any existing locks
		for(unsigned f = 0; f < raw_feat_frequency_creation_thread_lock.size(); ++f)
			omp_destroy_lock(&(raw_feat_frequency_creation_thread_lock[f]));
		raw_feat_frequency_creation_thread_lock.resize(num_raw_features);

		// Initialise locks for the raw features
		raw_features_valid.resize(num_raw_features);
		for(int f = 0; f < num_raw_features; ++f)
		{
			raw_features_valid[f] = false;
			omp_init_lock(&(raw_feat_frequency_creation_thread_lock[f]));
		}
	}

	// Memoiser for values calculated using the spatial method
	num_pixels = xsize*ysize;
	spat_memoiser.clear();
	spat_memoiser_valid.clear();
	if(use_memoiser)
	{
		spat_memoiser.resize(num_raw_features);
		spat_memoiser_valid.resize(num_raw_features);

		for(int f = 0; f < num_raw_features; ++f)
		{
			spat_memoiser[f].resize(num_pixels);
			spat_memoiser_valid[f].resize(num_pixels);
			std::fill(spat_memoiser_valid[f].begin(),spat_memoiser_valid[f].end(),false);
		}
	}

	// Compose lists of the possible output features
	createDerivedFeatureList(feature_set);

	// Create storage for raw feature magnitude images
	// Only need this if we have full images from frequency calculations
	raw_feature_magnitude_images.clear();
	// Destroy any existing locks
	for(unsigned f = 0; f < magnitude_images_creation_locks.size(); ++f)
		omp_destroy_lock(&(magnitude_images_creation_locks[f]));
	magnitude_image_valid.clear();
	if(use_frequency)
	{
		raw_feature_magnitude_images.resize(num_magnitude_features);
		// Create new locks
		magnitude_images_creation_locks.resize(num_magnitude_features);
		for(int f = 0; f < num_magnitude_features; ++f)
			omp_init_lock(&(magnitude_images_creation_locks[f]));
		magnitude_image_valid.resize(num_magnitude_features,false);
	}

	// Create storage fo coupled images
	// Destroy any existing locks
	coupled_image_valid.clear();
	coupled_images.clear();
	for(unsigned f = 0; f < coupled_images_creation_locks.size(); ++f)
		omp_destroy_lock(&(coupled_images_creation_locks[f]));
	if(use_vectorised_coupling)
	{
		// Create storage for coupled feature images
		coupled_images.resize(num_coupled_images);
		// Create new locks
		coupled_images_creation_locks.resize(num_coupled_images);
		for(int f = 0; f < num_coupled_images; ++f)
			omp_init_lock(&(coupled_images_creation_locks[f]));
		coupled_image_valid.resize(num_coupled_images,false);
	}

	// Create lists to count usage of features (to help decide the fastest way to perform calculations)
	raw_feat_usage_this_frame.clear();
	raw_feat_usage_last_frame.clear();
	if(auto_method)
	{
		raw_feat_usage_this_frame.resize(num_raw_features,0);
		raw_feat_usage_last_frame.resize(num_raw_features,0);
		// Set the list so that on the first frame, frequency domain calculations are
		// always used
		for(int f = 0; f < num_raw_features; ++f)
			raw_feat_usage_last_frame[f] = std::numeric_limits<int>::max() - 1;

		// Determine the threshold for using spatial convolutions over Fourier
		// domain calculations
		use_spatial_threshold = speedTestForThresh(); // number of points
	}
	else
	{
		use_spatial_threshold = 0;
		use_pixelwise_coupling_threshold = 0;
	}

	// Determine the threshold below which we should use elemtwise coupling
	// rather than vectorised coupling
	if( use_vectorised_coupling && (couple_method == comAuto) )
		use_pixelwise_coupling_threshold = speedTestForCoupling();
	else
		use_pixelwise_coupling_threshold = 0;

}

// Destructor
RIFeatExtractor::~RIFeatExtractor(void)
{
	// Destroy multi-threading locks
	for(unsigned i = 0; i < raw_feat_frequency_creation_thread_lock.size(); ++i)
		omp_destroy_lock(&(raw_feat_frequency_creation_thread_lock[i]));

	for(unsigned i = 0; i < magnitude_images_creation_locks.size(); ++i)
		omp_destroy_lock(&(magnitude_images_creation_locks[i]));

	for(unsigned i = 0; i < coupled_images_creation_locks.size(); ++i)
		omp_destroy_lock(&(coupled_images_creation_locks[i]));

}


// Create a list of the output features
// The list consists of three parts:
// The primary part indicates the indicates of the raw features the that output
// output feature is based on.
// The second part indicates a feature to couple the primary feature with,
// if none this is set to -1
// The third part describes the type of the resulting feature. Either magnitude,
// real, or imaginary
void RIFeatExtractor::createDerivedFeatureList(const featureSet_enum feature_set)
{
	// Reset the vectors we need
	derived_feature_primary_list.clear();
	derived_feature_secondary_list.clear();
	derived_feature_type_list.clear();

	raw_feat_to_magnitude_index.resize(num_raw_features);
	num_magnitude_features = 0;

	// Loop through the primary features once to do 'simple' features
	for( int f1 = 0; f1 < num_raw_features; ++f1)
	{
		// If this is a purely real feature - just extract the real part
		if((raw_feat_k_list[f1] == 0) && (raw_feat_m_list[f1] == 0))
		{
			derived_feature_primary_list.emplace_back(f1);
			derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
			derived_feature_type_list.emplace_back(ftReal);
			raw_feat_to_magnitude_index[f1] = -1;
		}
		// If it's a complex-valued rotation invariant feature, include
		// both parts
		else if(raw_feat_r_list[f1] == 0)
		{
			derived_feature_primary_list.emplace_back(f1);
			derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
			derived_feature_type_list.emplace_back(ftReal);
			derived_feature_primary_list.emplace_back(f1);
			derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
			derived_feature_type_list.emplace_back(ftImaginary);
			raw_feat_to_magnitude_index[f1] = -1;
		}
		// Otherwise, include just the absolute value
		else
		{
			derived_feature_primary_list.emplace_back(f1);
			derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
			derived_feature_type_list.emplace_back(ftMagnitude);
			raw_feat_to_magnitude_index[f1] = num_magnitude_features++;
		}
	} // f1 loop

	coupled_image_index_lookup.clear();
	num_coupled_images = 0;
	if(use_vectorised_coupling)
	{
		coupled_image_index_lookup.resize(num_raw_features);
		for(int f1 = 0; f1 < num_raw_features; ++f1)
		coupled_image_index_lookup[f1].resize(num_raw_features-f1-1,-1);
	}

	if(feature_set == fsSimpleCouple)
	{
		// Loop through again to do coupled features
		for( int f1 = 0; f1 < num_raw_features - 1; ++f1)
		{
			// Search f2 values for valid couples (start at f1 to avoid
			// duplicates)
			for( int f2 = f1 + 1; f2 < num_raw_features; ++f2)
			{
				// Two features can be coupled provided that r1 = r2 neq 0
				// both the real and imaginary parts of the resulting number
				// are rotation invariant
				if((raw_feat_r_list[f1] == raw_feat_r_list[f2]) && (raw_feat_r_list[f1] != 0))
				{
					derived_feature_primary_list.emplace_back(f1);
					derived_feature_secondary_list.emplace_back(f2);
					derived_feature_type_list.emplace_back(ftReal);
					derived_feature_primary_list.emplace_back(f1);
					derived_feature_secondary_list.emplace_back(f2);
					derived_feature_type_list.emplace_back(ftImaginary);
					if(use_vectorised_coupling)
						coupled_image_index_lookup[f1][f2-f1-1] = num_coupled_images++;
				}
			} // f2 loop
		} // f1 loop
	}
	else if(feature_set == fsExtraCouple)
	{
		// Loop through again to do coupled features
		for( int f1 = 0; f1 < num_raw_features - 1; ++f1)
		{
			// Search f2 values for valid couples (start at f1 to avoid
			// duplicates)
			for( int f2 = f1 + 1; f2 < num_raw_features; f2++)
			{
				// Two features can be coupled provided that r1 neq 0 and r2 neq 0
				// both the real and imaginary parts of the resulting number
				// are rotation invariant
				if((raw_feat_r_list[f1] != 0) && (raw_feat_r_list[f2] != 0))
				{
					derived_feature_primary_list.emplace_back(f1);
					derived_feature_secondary_list.emplace_back(f2);
					derived_feature_type_list.emplace_back(ftReal);
					derived_feature_primary_list.emplace_back(f1);
					derived_feature_secondary_list.emplace_back(f2);
					derived_feature_type_list.emplace_back(ftImaginary);
					if(use_vectorised_coupling)
						coupled_image_index_lookup[f1][f2-f1-1] = num_coupled_images++;
				}
			} // f2 loop
		} // f1 loop
	}

	// Store the length of the vector as a variable
	num_derived_features = derived_feature_primary_list.size();
}

// Creates the R and theta images that are later used to create the
// spatial basis functions
void RIFeatExtractor::findSpatSupport()
{
	int x,y;
	cv::MatIterator_<float> it_R, it_theta, end_R;

	// Calculate the size (round up to nearest odd number)
	spat_basis_size = std::ceil(2*basis_radius);
	if (spat_basis_size%2 == 0)
		++spat_basis_size;
	spat_basis_half_size = (spat_basis_size-1)/2;

	// Coordinate of the midpoint (both dimensions)
	const float midpoint = (spat_basis_size-1.0)/2.0;

	// Create the arrays
	R = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);
	theta = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);

	// Initialise loop variables
	it_theta = theta.begin();
	end_R = R.end();
	x = 0;
	y = 0;

	// Loop through pixels
	for(it_R = R.begin(); it_R != end_R; ++it_R,++it_theta)
	{
		// Coordinates relative to the centre of the image
		const float x_c = x - midpoint;
		const float y_c = midpoint - y; // change to up is positive definition

		// R and theta values
		*it_R = std::sqrt(std::pow(x_c,2) + std::pow(y_c,2));
		*it_theta = std::atan2(y_c,x_c);

		// Location of the next pixel
		++x;
		if (x == spat_basis_size)
		{
			x = 0;
			++y;
		}
	} // end pixel loop
}

// This function creates a single basis function in the spatial domain
void RIFeatExtractor::spatBasisSoftHist(const int u, const int j, const int k)
{
	cv::Mat_<float> radial_part, parts[2];

	// Find the radial part
	radial_part = 1.0 - cv::abs(R - j*r_space)/r_space;

	if(k == 0)
	{
		threshold(radial_part,parts[0],0.0,0.0,cv::THRESH_TOZERO);
		parts[1] = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);
	}
	else
	{
		// Set the centre to zero
		cv::MatIterator_<float> it = radial_part.begin();
		it += (spat_basis_size + 1)*spat_basis_half_size;
		it[0] = 0.0;

		threshold(radial_part,radial_part,0.0,0.0,cv::THRESH_TOZERO);
		polarToCart(radial_part,k*theta,parts[0],parts[1]);
	}

	merge(parts,2,U_spat[u]);

	// Normalise by radius squared to give area normalisation to the features
	U_spat[u] /= area_normaliser;

	// Cut out only the required central region
	const int reqhalfsize = std::ceil((j+1)*r_space) > spat_basis_half_size ? spat_basis_half_size : std::ceil((j+1)*r_space);
	U_spat[u] = U_spat[u](cv::Range(spat_basis_half_size-reqhalfsize,spat_basis_half_size+reqhalfsize+1),cv::Range(spat_basis_half_size-reqhalfsize,spat_basis_half_size+reqhalfsize+1));

}

// This function creates a single Zernike basis function in the spatial domain
// See Khotanzad and Hong 1990 "Invariant Image Recognition by Zernike Moments"
// for equations (NB j plays the role of n, k plays the role of m)
void RIFeatExtractor::spatBasisZernike(const int u, const int j, const int k)
{
	int s, absk, smax;
	float coef;
	cv::Mat radial, mask, parts[2], term;

	absk = std::abs(k);
	smax = (j - absk)/2;

	if(j == 0)
	{
		threshold(R,parts[0],basis_radius,1.0/area_normaliser,cv::THRESH_BINARY_INV);
		parts[1] = cv::Mat::zeros(spat_basis_size,spat_basis_size, CV_32F);
		merge(parts,2,U_spat[u]);
	}
	else
	{
		// Create a circular mask
		threshold(R,mask,basis_radius,1.0,cv::THRESH_TOZERO_INV);

		// Initialise to zero
		radial = cv::Mat::zeros(spat_basis_size,spat_basis_size, CV_32F);
		// Loop over the terms in the sum (indexed by s)
		for(s = 0; s <= smax; s++)
		{
			// Find the premultiplying coefficient for this term
			coef = factorial(j-s)/(factorial(s)*factorial((j+absk)/2-s)*factorial((j-absk)/2-s));

			if(j-2*s == 0)
				threshold(R,term,basis_radius,1.0,cv::THRESH_BINARY_INV);
			else
				cv::pow(mask/basis_radius,j-2*s,term);

			// Add this polynomial term to the sum
			if(s % 2 == 0)
				radial = radial + coef*term;
			else
				radial = radial - coef*term;
		}

		polarToCart(radial,k*theta,parts[0],parts[1]);
		merge(parts,2,U_spat[u]);

		// Normalise by radius squared to give area normalisation to the features
		U_spat[u] /= area_normaliser;
	}

}

// Simple factorial function
int RIFeatExtractor::factorial(const int n)
{
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

/*! \brief Calculate the value of a raw feature at a single image location.
*
* This method always uses the spatial (convolution) calculation method regardless
* of how the object was initialised. Ensure that an image has been placed into
*  the extractor before calling this method.
* \param raw_feature_num Index of the raw feature to calculate
* \param p Image location at which to calculate the feature.
* \return The raw feature value in complex form.
*/
std::complex<float> RIFeatExtractor::singleWindowFeature(const int raw_feature_num, const cv::Point p)
{
	// First check whether the value has already been calculated
	const int point_id = p.y*xsize + p.x;
	if(use_memoiser)
	{
		if(spat_memoiser_valid[raw_feature_num][point_id])
			return spat_memoiser[raw_feature_num][point_id];
	}

	cv::Mat_<cv::Vec2f> win, temp;
	cv::Scalar s;
	const int halfsize = (U_spat[raw_feat_basis_list[raw_feature_num]].rows-1)/2;
	const cv::Rect roi(p-cv::Point(halfsize,halfsize),p+cv::Point(halfsize+1,halfsize+1));

	if(nm == 0)
	{
		// Set win to the relevant subwindow of the input image
		win = I_2chan(roi);
	}
	else
	{
		// Set win to the relevant subwindow of the input image
		win = I_fou[raw_feat_m_list[raw_feature_num]](roi);
	}

	// Note that despite its name, mulSpectrums does nothing more than
	// element-wise complex multiplication, which is exactly what we
	// need here
	mulSpectrums(win,U_spat[raw_feat_basis_list[raw_feature_num]],temp,0);
	s = cv::sum(temp);
	const std::complex<float> result = std::complex<float>(s(0),s(1));

	// Put into the memoiser if using
	if(use_memoiser)
	{
		spat_memoiser[raw_feature_num][point_id] = result;
		spat_memoiser_valid[raw_feature_num][point_id] = true;
	}

	return result;
}

// Function to construct a frequency-domain rotation invariant basis function
void RIFeatExtractor::freqBasisSoftHist()
{
	const float xsizef = (float) pad_xsize;
	const float ysizef = (float) pad_ysize;
	cv::Mat_<float> psi;

	// Find coordinates at which we switch to negative frequencies
	const int xswitch = (pad_xsize % 2 == 0) ? pad_xsize/2 : (pad_xsize+1)/2;
	const int yswitch = (pad_ysize % 2 == 0) ? pad_ysize/2 : (pad_ysize+1)/2;

	// Construct an array of the angular coordinate of the frequency
	// domain
	psi = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
	for (int x = 0 ; x < pad_xsize; ++x)
	{
		for (int y = 0 ; y < pad_ysize; ++y)
		{
			if((x==0) && (y==0))
				psi(0,0) = 0.0;
			else
			{
				const float w_x = (x < xswitch) ? float(-x)/xsizef : (xsizef-float(x))/xsizef;
				const float w_y = (y < yswitch) ? float(y)/ysizef : (float(y)- ysizef)/ysizef;
				psi(y,x) = std::atan2(w_y,w_x);
			}
		}
	}

	// Loop through k values (rotation orders)
	#pragma omp parallel for
	for(int k = 0; k <= nk; ++k)
	{
		// Find the index of the basis with this value of k (and -k
		// where appropriate) and j = 0;
		int pos_k_basis_index = (nm == 0) ? k : nk + k;
		int neg_k_basis_index = (nm == 0) ? 0 : nk - k;

		// Create intermediate arrays, these will hold the Hankel transforms
		// of the cones used to construct the bases
		std::vector<cv::Mat_<float>> intermediate_array(nj);

		// Initialise arrays to zero
		for(int j = 0; j < nj; ++j)
		{
			intermediate_array[j] = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
		}

		// Loop through pixels to construct the intermediate arrays
		for (int x = 0 ; x < xswitch; ++x)
		{
			for (int y = 0 ; y < yswitch; ++y)
			{
				// Find freq value of this pixel location in polar coordinates
				// note both directions are flipped (*-1) relative to the definition used in the
				// image domain (up,right positive) to account for the flipping
				// involved in convolution
				const float w_x = float(-x)/xsizef ;
				const float w_y = float(y)/ysizef ;
				const float rho = 2.0*M_PI*std::sqrt(w_x*w_x + w_y*w_y);

				for(int j = 0; j < nj; ++j)
				{
					const float f = ((x == 0) && (y == 0)) ? 0.0 : coneHankel(rho,(j+1)*r_space,k)/area_normaliser;

					intermediate_array[j](y,x) = f; // topleft quadrant
					if(x > 0)
						intermediate_array[j](y,pad_xsize-x) = f; // topright quadrant
					if(y > 0)
						intermediate_array[j](pad_ysize-y,x) = f; // bottomleft quadrant
					if((x > 0) && (y > 0))
						intermediate_array[j](pad_ysize-y,pad_xsize-x) = f; // bottomright quadrant
				}
			} // y loop
		} // x loop

		// Use the intermediate arrays to contruct the full radial parts of
		// the filters
		for(int j = 0; j < nj; ++j)
		{
			cv::Mat_<float> complex_parts[2], radial_part;

			switch(j)
			{
				case 0:
					//intermediate_array[0].copyTo(radial_part);
					radial_part = 2*M_PI*intermediate_array[0];
					break;
				case 1:
					radial_part = 2*M_PI*(2*intermediate_array[1] - 2*intermediate_array[0]);
					break;
				default:
					radial_part = 2*M_PI*((j+1)*intermediate_array[j] - 2*j*intermediate_array[j-1] + (j-1)*intermediate_array[j-2]);
					break;
			}

			// Set the DC component
			if(k == 0)
			{
				// Set to the volume of the cone
				switch(j)
				{
					case 0:
						radial_part(0,0) = (1.0/3.0)*M_PI*std::pow(r_space,2.0);
						break;
					case 1:
						radial_part(0,0) = 2.0*(1.0/3.0)*M_PI*std::pow(2*r_space,2.0) - 2.0*(1.0/3.0)*M_PI*std::pow(r_space,2.0);
						break;
					default:
						radial_part(0,0)= (j+1)*(1.0/3.0)*M_PI*std::pow((j+1)*r_space,2.0) - 2.0*j*(1.0/3.0)*M_PI*std::pow(j*r_space,2.0) + (j-1)*(1.0/3.0)*M_PI*std::pow((j-1)*r_space,2.0);
						break;
				}
				// Normalise for area of the basis
				radial_part(0,0) = radial_part(0,0)/area_normaliser;

			} // else the values are already zero, as desired

			// Combine with the angular part and store in the relevant
			// element of the basis list
			polarToCart(radial_part,k*psi,complex_parts[0],complex_parts[1]);

			// Now need to multiply by i^(-k)
			switch(k%4)
			{
				case 0: // i^(-k) = 1
					// Nothing to do
					break;

				case 1: // i^(-k) = -i
				{
					cv::Mat_<float> temp = -complex_parts[0].clone();
					complex_parts[0] = complex_parts[1];
					complex_parts[1] = temp;
				}
					break;

				case 2: // i^(-k) = -1
					complex_parts[0] *= -1.0;
					complex_parts[1] *= -1.0;
					break;

				case 3: // i^(-k) = i
				{
					cv::Mat_<float> temp = complex_parts[0].clone();
					complex_parts[0] = -complex_parts[1];
					complex_parts[1] = temp;
				}
					break;
			}
			merge(complex_parts,2,U_freq[pos_k_basis_index]);

			// Same for negative k
			if(nm > 0)
			{
				// Need to flip about both frequency axes for negative k
				// But this is a bit awkward due to the construction of the spectrum
				// So let's be a bit smarter and use the symmetry of the basis functions

				// If k is even, then the real part of the is symmetric about both axes, so no need to flip
				// But the imaginary part is odd-symmetric, so multiply by -1 to flip
				if((k)%2 == 0)
					complex_parts[1] *= -1.0;
				// If k is odd, the opposite is true (due to swapping the two parts in the previous switch)
				// So multiply the real part by -1
				else
					complex_parts[0] *= -1.0;
				merge(complex_parts,2,U_freq[neg_k_basis_index]);
			}

			// Find the index to fill next time round the loop
			if(nm == 0)
				pos_k_basis_index += nk+1;
			else
			{
				pos_k_basis_index += 2*nk+1;
				neg_k_basis_index += 2*nk+1;
			}
		}
	} // end k loop
}

// Function to construct a frequency-domain rotation invariant basis function
void RIFeatExtractor::freqBasisZernike()
{
	const float xsizef = (float) pad_xsize;
	const float ysizef = (float) pad_ysize;
	cv::Mat_<float> psi;

	// Find coordinates at which we switch to negative frequencies
	const int xswitch = (pad_xsize % 2 == 0) ? pad_xsize/2 : (pad_xsize+1)/2;
	const int yswitch = (pad_ysize % 2 == 0) ? pad_ysize/2 : (pad_ysize+1)/2;

	// Construct an array of the angular coordinate of the frequency
	// domain
	psi = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
	for (int x = 0 ; x < pad_xsize; x++)
	{
		for (int y = 0 ; y < pad_ysize; y++)
		{
			if((x==0) && (y==0))
				psi(0,0) = 0.0;
			else
			{
				float w_x = (x < xswitch) ? float(-x)/xsizef : (xsizef-float(x))/xsizef;
				float w_y = (y < yswitch) ? float(y)/ysizef : (float(y)- ysizef)/ysizef;
				psi(y,x) = std::atan2(w_y,w_x);
			}
		}
	}

	#pragma omp parallel for
	for(int j = 0; j < nj; j++)
	{
		int basis_index;
		cv::Mat_<float> radial_part;

		// First index to fill for this radial order (j)
		if (nm == 0)
			basis_index = (j%2 == 0) ? j*(2+j)/4 : (j+1)*(j+1)/4;
		else
			basis_index = j*(j+1)/2;


		// Initialise the radial part of the spectrum for this radial order
		radial_part = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);

		// Loop through pixels to construct the radial part
		for (int x = 0 ; x < xswitch; ++x)
		{
			for (int y = 0 ; y < yswitch; ++y)
			{
				// Find freq value of this pixel location in polar coordinates
				// note both directions are flipped (*-1) relative to the definition used in the
				// image domain (up,right positive) to account for the flipping
				// involved in convolution
				const float w_x = float(-x)/xsizef ;
				const float w_y = float(y)/ysizef ;
				const float rho = 2.0*M_PI*std::sqrt(w_x*w_x + w_y*w_y);

				const float f = ((x == 0) && (y == 0)) ? 0.0 : boost::math::cyl_bessel_j(j+1,basis_radius*rho)/(rho*area_normaliser);

				radial_part(y,x) = f; // topleft quadrant
				if(x > 0)
					radial_part(y,pad_xsize-x) = f; // topright quadrant
				if(y > 0)
					radial_part(pad_ysize-y,x) = f; // bottomleft quadrant
				if((x > 0) && (y > 0))
					radial_part(pad_ysize-y,pad_xsize-x) = f; // bottomright quadrant

			} // y loop
		} // x loop

		// Now we have the radial part, cycle through the rotation orders
		for(int k = (nm == 0) ? j%2 : -j ; k <= j; k += 2)
		{
			cv::Mat_<float> complex_parts[2];

			// Combine with the angular part and store in the relevant
			// element of the basis list
			if(((j-std::abs(k))/2)%2 == 1)
				radial_part *= -1;

			// Set the DC component for j,k = 0
			if((k == 0) && (j == 0))
			{
				radial_part(0,0) = 1.0/(2.0*basis_radius);
			}

			// This switch does the slightly awkward business of multiplying by i^(-k)
			switch(std::abs(k)%4)
			{
				case 0: // * 1
				case 4:
					polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[0],complex_parts[1]);
					break;
				case 1: // * -i
					polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[1],complex_parts[0]);
					complex_parts[1] *= -1;
					break;
				case 2: // * -1
					polarToCart(-2*M_PI*basis_radius*radial_part,k*psi,complex_parts[0],complex_parts[1]);
					break;
				case 3: // * i
					polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[1],complex_parts[0]);
					complex_parts[0] *= -1;
					break;
			}
			// Account for negative k
			if((k < 0) && (std::abs(k)%2 == 1))
			{
				complex_parts[0] *= -1;
				complex_parts[1] *= -1;
			}
			merge(complex_parts,2,U_freq[basis_index]);

			basis_index += 1;

		} // end k loop

	} // end j loop

}

// Calculate the Hankel transform at frequency rho of a conical basis
// function of radius a and rotation order k
float RIFeatExtractor::coneHankel(const float rho, const float a, const int k)
{
	using boost::math::cyl_bessel_j;
	float h = 0.0;

	switch(std::abs(k))
	{
		case 0:
			h = 1.0/(a*std::pow(rho,3))*struvePhi(a*rho) ;
		break;
		case 1:
			h = struvePhi(rho*a)/std::pow(rho,2) - (2.0/std::pow(rho,2))*cyl_bessel_j(1,a*rho) + a*cyl_bessel_j(0,a*rho)/rho;
		break;
		case 2:
			h = (1.0/std::pow(rho,2))*cyl_bessel_j(0,a*rho) + (2.0/std::pow(rho,2)) - (3.0/(a*std::pow(rho,3)))*struveLambda0(a*rho);
		break;
		case 3:
			h = (8.0/(a*std::pow(rho,3)))*(cyl_bessel_j(0,a*rho) - 1.0) - (2.0/std::pow(rho,2))*cyl_bessel_j(1,a*rho) + (3.0/std::pow(rho,2))*struveLambda0(a*rho);
		break;
		case 4:
			h = -(1.0/std::pow(rho,2))*cyl_bessel_j(0,a*rho) + (24.0/(a*std::pow(rho,3)))*cyl_bessel_j(1,a*rho) + (4.0/std::pow(rho,2)) - 15.0/(a*std::pow(rho,3))*struveLambda0(a*rho);
		break;
		case 5:
			h = -(8.0/(a*std::pow(rho,3)))*cyl_bessel_j(0,a*rho) + (64.0/(std::pow(a,2)*std::pow(rho,4)) - 6.0/(std::pow(rho,2)))*cyl_bessel_j(1,a*rho) - (24.0/(a*std::pow(rho,3))) + 5.0/(std::pow(rho,2))*struveLambda0(a*rho);
		break;
		case 6:
			h = (1.0/std::pow(rho,2) - 160.0/(std::pow(a,2)*std::pow(rho,4)))*cyl_bessel_j(0,a*rho) + (16.0/(a*std::pow(rho,3)) + 320.0/(std::pow(a,3)*std::pow(rho,5)))*cyl_bessel_j(1,a*rho) + (6.0/std::pow(rho,2)) - 35.0/(a*std::pow(rho,3))*struveLambda0(a*rho);
		break;
	} // end switch

	return h;
}

// Helper functions for coneHankel
float RIFeatExtractor::struvePhi(const float x)
{
	return 0.5*M_PI*x*(boost::math::cyl_bessel_j(1,x)*struveh0(x) - boost::math::cyl_bessel_j(0,x)*struveh1(x));
}

// Helper functions for coneHankel
float RIFeatExtractor::struveLambda0(const float x)
{
	return x*boost::math::cyl_bessel_j(0,x) + struvePhi(x);
}

// Perform a speed test to determine for what number of input pixels it
// becomes more efficient to use the Fourier domain calculation to evaluate
// one raw feature
int RIFeatExtractor::speedTestForThresh()
{
	cv::Mat_<float> test_image, test_image_ori;
	double t, time_taken_spat, time_taken_freq;
	std::complex<float> out;
	cv::Vec2f vec_result;

	// Create a blank test image
	if(nm == 0)
	{
		test_image = cv::Mat::ones(ysize,xsize,CV_32F);
		setScalarInputImage(test_image);
	}
	else
	{
		test_image = cv::Mat::ones(ysize,xsize,CV_32F);
		test_image_ori = cv::Mat::ones(ysize,xsize,CV_32F);
		setVectorInputImage(test_image,test_image_ori);
	}

	// Run the spatial filter several times
	t = (double)cv::getTickCount();
	for(int i = 0 ; i < C_SPEED_TEST_RUNS; ++i)
	{
		out = singleWindowFeature(i%num_raw_features,cv::Point(xsize/2, ysize/2));
		dummy_result_float = out.real(); // ensure the loop isn't optimised away
	}
	time_taken_spat = ((double)cv::getTickCount() - t)/((double)cv::getTickFrequency());

	// Run the Fourier domain filter several times
	t = (double)cv::getTickCount();
	for(int i = 0 ; i < C_SPEED_TEST_RUNS; ++i)
	{
		rawFeatureFrequencyCalculation(i%(num_bases),nm);
		vec_result = raw_feats_unpadded[index_lookup_table[nm*num_bases+i%num_bases]].at<cv::Vec2f>(cv::Point(0,0));
		dummy_result_float = vec_result[0]; // ensure the loop isn't optimised away
	}
	time_taken_freq = ((double)cv::getTickCount() - t)/((double)cv::getTickFrequency());

	return std::round(time_taken_freq/time_taken_spat);
}

// Work out how many single coupling calculations it takes before it's quicker to vectorise
// for the entire image
int RIFeatExtractor::speedTestForCoupling()
{
	// Timing variables
	double t, time_taken_mat, time_taken_elem;

	// Test coupling the whole image in one go
	cv::Mat_<cv::Vec2f> test_image_1, test_image_2;
	cv::Mat_<float> temp_planes[2];
	test_image_1 = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
	test_image_2 = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
	cv::randu(test_image_1,cv::Vec2f(0.0,0.0),cv::Vec2f(100.0,100.0));
	cv::randu(test_image_2,cv::Vec2f(0.0,0.0),cv::Vec2f(100.0,100.0));

	t = (double) cv::getTickCount();
	for(int i = 0; i < C_COUPLING_SPEED_TEST_NUM; ++i)
	{
		cv::Mat_<cv::Vec2f> result;
		cv::Mat_<cv::Vec2f> coupled_unnormalised;
		cv::mulSpectrums(test_image_1,test_image_2,coupled_unnormalised,0,true);
		cv::split(coupled_unnormalised,temp_planes);
		// Temporarily store the magnitude in temp
		cv::Mat_<float> mag;
		cv::magnitude(temp_planes[0], temp_planes[1], mag);
		// Normalise the real part
		cv::divide(temp_planes[0],mag,temp_planes[0]);
		cv::divide(temp_planes[1],mag,temp_planes[1]);
		cv::merge(temp_planes,2,coupled_images[0]);
	}
	time_taken_mat = (((double)cv::getTickCount() - t)/((double)cv::getTickFrequency()))/C_COUPLING_SPEED_TEST_NUM;

	// Test coupling each element singly
	t = (double) cv::getTickCount();
	for(int i = 0; i < C_COUPLING_SPEED_TEST_NUM; ++i)
	{
		float* const f1_val = test_image_1.ptr<float>(i%pad_ysize,i%pad_xsize);
		const std::complex<float> c1 = std::complex<float>(f1_val[0],f1_val[1]);
		float* const f2_val = test_image_2.ptr<float>((i+100)%pad_ysize,(i+100)%pad_xsize);
		const std::complex<float>c2 = std::complex<float>(f2_val[0],f2_val[1]);
		const std::complex<float> complex_result = c1*std::conj(c2);
		dummy_result_float = complex_result.real()/std::abs(complex_result);
	}
	time_taken_elem = (((double)cv::getTickCount() - t)/((double)cv::getTickFrequency()))/C_COUPLING_SPEED_TEST_NUM;

	return std::round(time_taken_mat/time_taken_elem);
}

/*! \brief Call this function to put a new scalar input image into the object.
*
* Invalidates any data held on the previous image and performs preparatory operations
* on the new image. Use this function to put a scalar (single channel) image (such
* as a plain monochrome intensity image) into the extractor. Only call this
* if the object was initialised with num_fourier_coefs (m) = 0.
* \param in New input image. The object will take a local deep copy of this image.
*/
void RIFeatExtractor::setScalarInputImage(const cv::Mat &in)
{
	cv::Mat_<float> temp[2];
	cv::Mat_<cv::Vec2f> padded;

	// Set internal copy of input
	if(I.channels() > 1)
	{
		cv::Mat one_chan;
		cvtColor(in,one_chan,cv::COLOR_BGR2GRAY);
		one_chan.convertTo(I,CV_32F);
	}
	else
		in.convertTo(I,CV_32F);

	// Form a 2-channel version for faster spatial-domain calculations
	temp[0] = I;
	temp[1] = cv::Mat::zeros(I.rows,I.cols, CV_32F);
	merge(temp,2,I_2chan);

	// Pad and Fourier transform
	if(use_frequency)
	{
		copyMakeBorder(I, planes[0], 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));  //expand input image to optimal size
		merge(planes, 2, padded);

		// Find FFT
		dft(padded,FFT_im[0]);
	}

	refreshImage();

}

/*! \brief Call this function to put a new scalar input image into the object.
*
* Invalidates any data held on the previous image and performs preparatory operations
* on the new image. Use this function to put a vector (two channel) image (such
* as a gradient or motion field image) into the extractor.
* \param in_magnitude Magnitude of the new input image. The object will take a local deep copy of this image.
* \param in_orientation Orientationof the new input image. This must be defined
* in the standard mathematical sense, i.e. measured anti-clockwise from the positive
* x-axis (pointing right). The object will take a local deep copy of this image.
*/
void RIFeatExtractor::setVectorInputImage(const cv::Mat &in_magnitude, const cv::Mat &in_orientation)
{
	// Set the internal copies of the input
	in_magnitude.convertTo(I,CV_32FC1);
	in_orientation.convertTo(I_ori,CV_32FC1);

	// Expand into a set of Fourier coefficients
	expandFourierImages();

	refreshImage();
}

// Common code to execute when the image changes
void RIFeatExtractor::refreshImage()
{
	// Swap pointers to the feature usage counters
	raw_feat_usage_last_frame.swap(raw_feat_usage_this_frame);

	// Clear any data the object holds on the previous image
	// Mark all previously calculated raw featurse as invalid
	std::fill(raw_feat_usage_this_frame.begin(),raw_feat_usage_this_frame.end(),0);
	std::fill(raw_features_valid.begin(),raw_features_valid.end(),false);
	std::fill(magnitude_image_valid.begin(),magnitude_image_valid.end(),false);
	std::fill(coupled_image_valid.begin(),coupled_image_valid.end(),false);

	if(use_memoiser)
		for(int f = 0; f < num_raw_features; ++f)
			std::fill(spat_memoiser_valid[f].begin(),spat_memoiser_valid[f].end(),false);
}

// This function takes in a complex-valued (vector) image and expands
// each pixel into its Fourier series coefficients
void RIFeatExtractor::expandFourierImages()
{
	cv::Mat_<cv::Vec2f> padded;
	cv::Mat_<float> temp_planes[2]; // temporary memory for two image planes

	// Allocate memory for output if we have not already done so
	if(I_fou.size() == 0)
		I_fou.resize(nm+1); // memory for output


	// The zeroth coefficient (m=0) can be found simply from the magnitude
	I.copyTo(temp_planes[0]);
	temp_planes[1] = cv::Mat_<float>::zeros(I.rows,I.cols);
	merge(temp_planes,2,I_fou[0]);

	if(use_frequency)
	{
		cv::Mat_<cv::Vec2f> padded;

		// Pad the input array and Fourier transform
		if((I_fou[0].rows != pad_ysize) || (I_fou[0].cols != pad_xsize))
		{
			copyMakeBorder(I_fou[0], padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));
			dft(padded,FFT_im[0]);
		}
		else
			dft(I_fou[0],FFT_im[0]);
	}

	// Loop to find the other coefficients
	#pragma omp parallel for
	for(int m = 1; m <= nm; ++m)
	{
		cv::Mat_<cv::Vec2f> padded;
		cv::Mat_<float> temp_planes_thread[2]; // temporary memory for two image planes

		// Multiply the orientation by -m and combine with the magnitude
		// to give a Cartesian representation
		polarToCart(I,-m*I_ori,temp_planes_thread[0],temp_planes_thread[1]);

		// Merge this Cartesian representation into one complex-valued,
		// as required for frequency-domain filtering via the DFT
		merge(temp_planes_thread,2,I_fou[m]);

		if(use_frequency)
		{
			// Pad the input array and Fourier transform
			if((I_fou[m].rows != pad_ysize) || (I_fou[m].cols != pad_xsize))
			{
				copyMakeBorder(I_fou[m], padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));
				dft(padded,FFT_im[m]);
			}
			else
				dft(I_fou[m],FFT_im[m]);
		}
	}
}

/*! \brief Calculate all raw features
*
* Calculates all raw features and stores them internally ready for further queries.
* Note that it is not necessary to call this function in order to use other methods
* that make use of the raw features. However, if you know that you need every
* raw feature at a large number of image locations, this is the simplest, and
* probably the most efficient, way to do it.
*/
void RIFeatExtractor::createRawFeats()
{
	// Loop through histogram coefficients
	for(int m = 0; m <= nm; ++m)
	{
		// Loop through the bases
		#pragma omp parallel for
		for(int u = 0; u < num_bases; ++u)
		{
			// Skip negative rotation orders for m=0 to avoid duplication
			if((m == 0) && (basis_k_list[u] < 0)) continue;
			// Skip if the resulting rotation order is too high
			if((max_r >= 0) && std::abs(basis_k_list[u] - m) > max_r) continue;

			// Perform the frequency domain convolution
			rawFeatureFrequencyCalculation(u,m);
		}
	}
}

// This function is used to calculate a single raw feature for every pixel
// in the image using Fourier domain multiplication followed by inverse FFT
void RIFeatExtractor::rawFeatureFrequencyCalculation(const int f, const int u, const int m)
{
	cv::Mat_<cv::Vec2f> temp;

	// Perform frequency domain filtering and store in the raw_feat_images
	// array
	cv::mulSpectrums(FFT_im[m],U_freq[u],temp,0);
	cv::idft(temp, raw_feat_images[f],cv::DFT_SCALE);

	// Set the 'unpadded' Mat to the relevant part of the image
	raw_feats_unpadded[f] = raw_feat_images[f](cv::Range(0,ysize),cv::Range(0,xsize));
	raw_features_valid[f] = true;
}

// Overloaded version where the raw feature index f is looked up for you
void RIFeatExtractor::rawFeatureFrequencyCalculation(const int u, const int m)
{
	rawFeatureFrequencyCalculation(index_lookup_table[m*num_bases+u],u,m);
}

// Overloaded version where the basis index u and the m value are looked up for you
void RIFeatExtractor::rawFeatureFrequencyCalculation(const int f)
{
	rawFeatureFrequencyCalculation(f,raw_feat_basis_list[f],raw_feat_m_list[f]);
}

void RIFeatExtractor::raiseComplexImageToPower(const cv::Mat_<cv::Vec2f>& in, cv::Mat_<cv::Vec2f>& out, const int power)
{
	// Easiest to do this with polar representation
	cv::Mat_<float> mag, arg;
	cv::Mat_<float> temp_planes[2];
	cv::split(in,temp_planes);
	cv::cartToPolar(temp_planes[0],temp_planes[1],mag,arg);
	cv::pow(mag,power,mag);
	cv::polarToCart(mag,power*arg,temp_planes[0],temp_planes[1]);
	cv::merge(temp_planes,2,out);
}

void RIFeatExtractor::fullImageCouple(cv::Mat_<cv::Vec2f>& coupled_image, const int f1, const int f2)
{
	cv::Mat_<cv::Vec2f> coupled_unnormalised;
	cv::Mat_<float> temp_planes[2];

	// Pointwise complex multiplacation, conjugating the second feature
	if(raw_feat_r_list[f1] != raw_feat_r_list[f2])
	{
		cv::Mat_<cv::Vec2f> f1_power_image, f2_power_image;

		// Need to raise the images to the relevant power before multiplying
		if(raw_feat_r_list[f2] == 1)
			f1_power_image = raw_feats_unpadded[f1];
		else
			raiseComplexImageToPower(raw_feats_unpadded[f1],f1_power_image,raw_feat_r_list[f2]);

		if(raw_feat_r_list[f1] == 1)
			f2_power_image = raw_feats_unpadded[f2];
		else
			raiseComplexImageToPower(raw_feats_unpadded[f2],f2_power_image,raw_feat_r_list[f1]);

		mulSpectrums(f1_power_image,f2_power_image,coupled_unnormalised,0,true);
	}
	else // straightforward multiplication (conjugating second argument)
		mulSpectrums(raw_feats_unpadded[f1],raw_feats_unpadded[f2],coupled_unnormalised,0,true);

	// Now need to normalise by the magnitude
	split(coupled_unnormalised,temp_planes);

	// Find the magnitude
	cv::Mat_<float> mag;
	cv::magnitude(temp_planes[0], temp_planes[1], mag);
	// Normalise the coupled image
	cv::divide(temp_planes[0],mag,temp_planes[0]);
	cv::divide(temp_planes[1],mag,temp_planes[1]);
	cv::merge(temp_planes,2,coupled_image);

}

// Calculates a derived feature that depends on a single raw feature
float RIFeatExtractor::derivedFeatureFromComplex(const std::complex<float> complex_feat, const featureType_enum type)
{
	float result = 0.0;
	switch(type)
	{
		case ftMagnitude:
			assert(!std::isnan(std::abs(complex_feat)));
			result = std::abs(complex_feat);
		break;

		case ftReal:
			assert(!std::isnan(complex_feat.real()));
			result = complex_feat.real();
		break;

		case ftImaginary:
			assert(!std::isnan(complex_feat.real()));
			result = complex_feat.imag();
		break;
	}
	return result;
}

// Takes two raw features and returns the derived feature due to coupling
std::complex<float> RIFeatExtractor::coupleFeatures(std::complex<float> f1_val, const int r1, std::complex<float> f2_val, const int r2)
{
	// Normalise the features before multiplying to avoid numerical issues
	// Need to be careful about dividing by zero here
	// Arbitrarily set to 1.0 + 0.0 if the magnitude is too small
	// to divide by, this effectlively pick an arbitrary orientation
	if(std::abs(f1_val) < 1.0/std::numeric_limits<float>::max())
		f1_val = std::complex<float>(1.0,0.0);
	else
		f1_val /= std::abs(f1_val);

	if(std::abs(f2_val) < 1.0/std::numeric_limits<float>::max())
		f2_val = std::complex<float>(1.0,0.0);
	else
		f2_val /= std::abs(f2_val);

	// May need to raise the raw features to a power
	if(r1 != r2)
	{
		f2_val = std::pow(f2_val,r1);
		f1_val = std::pow(f1_val,r2);
	}

	const std::complex<float> coupledcmplx = f1_val*std::conj(f2_val);
	assert(!std::isnan(coupledcmplx.real()) && !std::isnan(coupledcmplx.imag()));

	return coupledcmplx;
}

// Check the validity of a raw feature image in a thread-safe manner and
// recalculate if required.
bool RIFeatExtractor::checkRawFeatureValidity(const int f, const bool calculate_if_invalid)
{
	if(!use_frequency)
		return false;

	bool valid;

	omp_set_lock(&(raw_feat_frequency_creation_thread_lock[f]));
	if(raw_features_valid[f])
	{
		valid = true;
	}
	else
	{
		if(calculate_if_invalid)
		{
			rawFeatureFrequencyCalculation(f);
			valid = true;
		}
		else
			valid = false;
	}

	omp_unset_lock(&(raw_feat_frequency_creation_thread_lock[f]));
	return valid;
}

// Check the validity of a coupled image in a thread-safe manner and
// recalculate if required.
bool RIFeatExtractor::checkCoupledImageValidity(const int f1, const int f2, const bool calculate_if_invalid, int& index)
{
	bool valid;
	assert(f2 > f1);
	index = coupled_image_index_lookup[f1][f2-f1-1];
	assert(index >= 0);

	omp_set_lock(&(coupled_images_creation_locks[index]));
	if(coupled_image_valid[index])
	{
		valid = true;
	}
	else
	{
		if(calculate_if_invalid)
		{
			fullImageCouple(coupled_images[index],f1,f2);
			coupled_image_valid[index] = true;
			valid = true;
		}
		else
			valid = false;
	}

	omp_unset_lock(&(coupled_images_creation_locks[index]));
	return valid;
}

// Ensure the validity of a magnitude image in a thread-safe manner get the associated
// index
int RIFeatExtractor::ensureMagnitudeImageValidity(const int f)
{
	const int i = raw_feat_to_magnitude_index[f];
	assert(i >= 0);
	omp_set_lock(&(magnitude_images_creation_locks[i]));

	if(!magnitude_image_valid[i])
	{
		cv::Mat_<float> temp_planes[2];
		cv::split(raw_feats_unpadded[f],temp_planes);
		cv::magnitude(temp_planes[0],temp_planes[1],raw_feature_magnitude_images[i]);
		magnitude_image_valid[i] = true;
	}

	omp_unset_lock(&(magnitude_images_creation_locks[i]));
	return i;
}


// Simple getter for the maximum output feature index that can be used
/*! \brief Get the number of derived features available in the current feature set
*
* \return The number of derived features
*/
int RIFeatExtractor::getNumDerivedFeats() const
{
	return num_derived_features;
}

// Simple getter for the maximum output feature index that can be used
/*! \brief Get the number of raw features available in the current feature set
*
* \return The number of raw features
*/
int RIFeatExtractor::getNumRawFeats() const
{
	return num_raw_features;
}

// For a given effective rotation order, r, this function returns a list (raw_feat_ind)
// of all the raw feature indices that have this effective rotation order. The length
// of the list is returned in num_feats
/*! \brief Get a list of raw feature indices with a given effective rotation order.
*
* For a given effective rotation order (r) get a list of indices of the raw features
* that have this effective rotation order.
* \param r The effective rotation order.
* \param raw_feat_ind This vector is returned by reference containing the list of raw
* feature indices that have effective rotation order r.
* \param include_negatives If true, features with a rotation order of -r are also
* included in the list.
* \param Jmax Only include raw features with that use a small basis function with
* \f$ (j < j_{max}\f$). Should be between 0 (inclusive) and the num_radii value
* that the object was constructed with (exclusive). If negative or omitted, raw
* features using all radial profiles are inlcuded.
*/
void RIFeatExtractor::getFeatsWithGivenR(const int r, std::vector<int>& raw_feat_ind, const bool include_negatives, const int Jmax) const
{
	// Clear any existing contents of the vector
	raw_feat_ind.clear();

	// Count the faw features with this effective rotation order
	for(int f = 0; f < num_raw_features; ++f)
		if( ( (raw_feat_r_list[f] == r) || (include_negatives && (raw_feat_r_list[f] == -r) ) ) && ( (Jmax < 0) || (raw_feat_j_list[f] <= Jmax) ) )
			raw_feat_ind.emplace_back(f);
}


/*! \brief Get a list of the derived features that use only small basis functions
*
* Finds a list of the derived features that can be calculated using only small basis functions (i.e.
* those with a small \f$ j < j_{max}\f$). This can be useful for training models
* that should use a smaller spatial support than the full detection radius.
* A vector of the indices of the derived features is returned by reference.
* \param Jmax Index of the maximum radial profile to to be included in the list.
* Should be between 0 (inclusive) and the num_radii value that the object was
* constructed with (exclusive).
* \param output_feature_list This is returned by reference and contains the list
* of derived feature indices that satisfy the condition.
*/
void RIFeatExtractor::getFeatsUsingLowJ(const int Jmax, std::vector<int>& output_feature_list) const
{
	// Make sure the vector is empty
	output_feature_list.clear();

	// Loop over all the output features and add them to the vector if they are valid
	for(int f = 0; f < num_derived_features; ++f)
	{
		const int rf1 = derived_feature_primary_list[f];
		const int rf2 = derived_feature_secondary_list[f];

		// Check that we can calculate the first raw feature with j <= Jmax
		if(raw_feat_j_list[rf1] > Jmax)
			continue;

		// Check the second feature
		if( (rf2 != C_SECOND_FEATURE_NONE) && (raw_feat_j_list[rf2] > Jmax ))
			continue;

		// If we got here, the feature is valid, add to the list
		output_feature_list.emplace_back(f);
	}
}

// Function to return the halfsize of the maximum basis used if features up to Jmax
// are used. If Jmax is negative or omitted, all features are assumed to be used
/*! \brief Get the 'halfsize' of the largest spatial basis function.
*
* The size of the spatial basis functions are always odd, and the 'halfsize' is
* value such that the side length of the square basis function image is 2*halfsize + 1.
* This is important because you must ensure that whenever you ask for a feature
* an image location, that location must be at least 'halfsize' pixels from the edge
* of the image.
* \param Jmax If this value is non-negative, only the basis functions with \f$j <= J_{max}\f$
* are considered. If only features using these basis functions are used, the image
* locations may be closer to the edge of the image than the halfsize of the biggest
* basis in the set. If Jmax is negative or not provided, all basis functions are
* considered.
* \return The halfsize of the largest basis function in the set.
*/
int RIFeatExtractor::getMaxSpatBasisHalfsize(const int Jmax) const
{
	if(Jmax < 0)
		return spat_basis_half_size;
	else
	{
		// Find the first basis with j = Jmax and return its halfsize
		int u = 0;
		while(basis_j_list[u] != Jmax) ++u;

		return (U_spat[u].cols-1)/2;
	}
}

/*! \brief Get a copy of one of the spatial basis functions images.
*
* \param basis_index Index of the basis function in question. Should be between 0
* (inclusive) and \c getNumBases() (exclusive).
* \return A deep copy (clone) of the requested basis function.
*/
cv::Mat RIFeatExtractor::getSpatialBasisCopy(const int basis_index) const
{
	// Check the requested basis exists
	if((basis_index >= 0) && (basis_index < num_bases))
		return U_spat[basis_index].clone();

	// Return empy matrix to signal faliure
	return cv::Mat();
}

/*! \brief Get a copy of the frequency domain representation of one of the
* basis functions.
*
* Do not call this if the object was initialised with the \c cmSpatial calculation
* method.
* \param basis_index Index of the basis function in question. Should be between 0
* (inclusive) and \c getNumBases() (exclusive).
* \return A deep copy (clone) of the frequency domain representation of the requested
* basis function.
*/
cv::Mat RIFeatExtractor::getFrequencyBasisCopy(const int basis_index) const
{
	// Check the requested basis exists
	if((basis_index >= 0) && (basis_index < num_bases))
		return U_freq[basis_index].clone();

	// Return empy matrix to signal faliure
	return cv::Mat();
}

/*! \brief Get the number of basis functions used by the object.
*
* \return The number of basis functions used.
*/
int RIFeatExtractor::getNumBases() const
{
	return num_bases;
}

/*! \brief Get the j and k parameters for a given basis in the list
*
* \param basis_index Index of the basis functionin question. Should be between 0
* (inclusive) and \c getNumBases() (exclusive).
* \param j The j value (radial profile index) of this basis function returned by reference.
* \param k The k value (rotation order) of this basis function returned by reference.
*/
bool RIFeatExtractor::getBasisInfo(const int basis_index, int& j, int& k) const
{
	// Check the requested basis exists
	if((basis_index >= 0) && (basis_index < num_bases))
	{
		j = basis_j_list[basis_index];
		k = basis_k_list[basis_index];
		return true;
	}

	// Otherwise return false with j=k=-1 to signal failure
	j = -1;
	k= -1;
	return false;
}

/*! \brief Converts a human readable string to the enumeration representing the
* calculation method.
*
* The list of accepted strings is:
* \li \c "spatial", or \c "s" is converted to \c cmSpatial
* \li \c "frequency", or \c "f" is converted to \c cmFrequency
* \li \c "auto", or \c "a" is converted to \c cmAuto
* \param method_string The string to convert into a calculation method
* \param method_enum The calculation method represented in enumeration form.
* \return True if conversion was succesful, otherwise false.
*/
bool RIFeatExtractor::stringToCalcMethod(const std::string& method_string, calculationMethod_enum& method_enum)
{
	if(method_string == "spatial" || method_string == "s")
	{
		method_enum = cmSpatial;
		return true;
	}
	if(method_string == "frequency" || method_string == "f")
	{
		method_enum = cmFrequency;
		return true;
	}
	if(method_string == "auto" || method_string == "a")
	{
		method_enum = cmAuto;
		return true;
	}
	return false;
}

/*! \brief Converts a human readable string to the enumeration representing the
* coupling method.
*
* The list of accepted strings is:
* \li \c "element-wise", or \c "e" is converted to \c comElementwise.
* \li \c "vectorised", or \c "v" is converted to \c comVectorised.
* \li \c "auto", or \c "a" is converted to \c comAuto.
* \param method_string The string to convert into a coupling method.
* \param method_enum The coupling method represented in enumeration form.
* \return True if conversion was succesful, otherwise false.
*/
bool RIFeatExtractor::stringToCoupleMethod(const std::string& method_string, couplingMethod_enum& method_enum)
{
	if(method_string == "element-wise" || method_string == "e")
	{
		method_enum = comElementwise;
		return true;
	}
	if(method_string == "vectorised" || method_string == "v")
	{
		method_enum = comVectorised;
		return true;
	}
	if(method_string == "auto" || method_string == "a")
	{
		method_enum = comAuto;
		return true;
	}
	return false;
}

/*! \brief Converts a human readable string to the enumeration representing the
* basis type.
*
* The list of accepted strings is:
* \li \c "softhist", \c "s", or \c "soft_histograms" is converted to \c btSoftHist.
* \li \c "zernike", or \c "z" is converted to \c btZernike.
* \param basis_type_string The string to convert into a basis type.
* \param basis_type_enum The basis type represented in enumeration form.
* \return True if conversion was succesful, otherwise false.
*/
bool RIFeatExtractor::stringBasisType(const std::string& basis_type_string, basisType_enum& basis_type_enum)
{
	if(basis_type_string == "softhist" || basis_type_string == "s" || basis_type_string == "soft_histograms")
	{
		basis_type_enum = btSoftHist;
		return true;
	}
	if(basis_type_string == "zernike" || basis_type_string == "z")
	{
		basis_type_enum = btZernike;
		return true;
	}
	return false;
}

/*! \brief Converts a human readable string to the enumeration representing the
* feature set type.
*
* The list of accepted strings is:
* \li \c "basic", or \c "b" is converted to \c fsBasic.
* \li \c "couple_simple", \c "simple", or \c "s" is converted to \c fsSimpleCouple.
* \li \c "couple_extra", \c "extra", or \c "e" is converted to \c fsExtraCouple.
* \param feature_set_string The string to convert into a feature set type.
* \param feature_set_enum The feature set type represented in enumeration form.
* \return True if conversion was succesful, otherwise false.
*/
bool RIFeatExtractor::stringToFeatureSet(const std::string& feature_set_string, featureSet_enum& feature_set_enum)
{
	if(feature_set_string == "basic" || feature_set_string == "b")
	{
		feature_set_enum = fsBasic;
		return true;
	}
	if(feature_set_string == "couple_simple" || feature_set_string == "simple" || feature_set_string == "c")
	{
		feature_set_enum = fsSimpleCouple;
		return true;
	}
	if(feature_set_string == "couple_extra" || feature_set_string == "extra" || feature_set_string == "ce")
	{
		feature_set_enum = fsExtraCouple;
		return true;
	}
	return false;

}

} // end of namespace RIFeatures
