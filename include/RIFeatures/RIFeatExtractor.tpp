// Written by Christopher Bridge for the RIFeatures library. Licensed under the
// GNU general public license version 3.

namespace RIFeatures
{

/*! \brief Get the value of a single derived feature from a number of image locations
*
* Takes a sequence of image locations (each described by a cv::Point) and calculates
* the specified derived feature at these image locations in the input image.
* The input image must have been set up prior to calling this method. The resulting
* feature values are placed into the corresponding elements of the output sequence.
*
* You must also ensure that all the points are valid, i.e. at least the
* 'halfsize' of the relevant basis function away from the image edge (see
* \c getMaxSpatBasisHalfsize()).
*
* This method is thread-safe in that it can be used in OpenMP parallel sections
* where other threads are calling this method other thread-safe methods without
* causing data races.
* \tparam TInputIterator The type of the iterator to the start of the input sequence of image locations. This must dereference to a cv::Point and be at least an input iterator (according to C++ standard library definitions).
* \tparam TOutputIterator The type of the iterator the output feature values. Must be at least an output iterator (according to C++ standard library definitions) and support assignment to a float value.
* \param first_point Iterator to the first image location, i.e. the centre of the first detection window.
* \param last_point Iterator to the end of the image location lists
* \param derived_feat_num The number of the derived feature to calculate (must be between 0 (inclusive) and getNumDerivedFeats() (exclusive))
* \param dest Output iterator to the start of the output sequence where the derived feature scores will be placed.
*/
template<typename TInputIterator,typename TOutputIterator>
void RIFeatExtractor::getDerivedFeature(TInputIterator first_point, const TInputIterator last_point, const int derived_feat_num, TOutputIterator dest)
{
	const int f1 = derived_feature_primary_list[derived_feat_num];
	const int f2 = derived_feature_secondary_list[derived_feat_num];
	const featureType_enum type = derived_feature_type_list[derived_feat_num];
	const int num_points = std::distance(first_point,last_point);

	// Update usage features for the current frame
	if(auto_method)
	{
		raw_feat_usage_this_frame[f1] += num_points;
		if(f2 != C_SECOND_FEATURE_NONE)
			raw_feat_usage_this_frame[f2] += num_points;
	}

	// Check whether we have already calculated the necessary raw feature
	// and calculate it if desired
	const bool f1_frequency_calculation_if_invalid = always_use_frequency || (auto_method && ((raw_feat_usage_last_frame[f1] > use_spatial_threshold) || (num_points > use_spatial_threshold)));
	const bool f1_image_valid = checkRawFeatureValidity(f1, f1_frequency_calculation_if_invalid);

	// Get the entire image of features for the first feature
	if((f2 == C_SECOND_FEATURE_NONE) && f1_image_valid)
	{
		getDerivedFromSingleRawFeature(f1,type,first_point,last_point,dest);
		return;
	}

	// Same for the secondary feature, if applicable
	const bool f2_frequency_calculation_if_invalid = (f2 == C_SECOND_FEATURE_NONE) ?
		false :
		always_use_frequency || (auto_method && ((raw_feat_usage_last_frame[f2] > use_spatial_threshold || num_points > use_spatial_threshold))) ;
	const bool f2_image_valid = (f2 == C_SECOND_FEATURE_NONE) ?
		false :
		checkRawFeatureValidity(f2,f2_frequency_calculation_if_invalid);

	// If we have entire images of both f1 and f2, we can vectorise the
	// coupling calculation
	if(f1_image_valid && f2_image_valid && use_vectorised_coupling)
	{
		int coupled_image_index;
		const bool full_image_couple_if_invalid = always_use_vectorised_coupling || ((num_points > use_pixelwise_coupling_threshold));

		if(checkCoupledImageValidity(f1,f2,full_image_couple_if_invalid,coupled_image_index))
		{
			// Loop through all the required points and just take points
			// straight out of feat_image
			switch(type)
			{
				case ftReal:
					while(first_point != last_point)
					{
						// Set the value in the output array
						const cv::Vec2f& element = coupled_images[coupled_image_index].at<cv::Vec2f>(*first_point++);
						assert(!std::isnan(element[0]));
						*dest++ = element[0];
					}
				break;
				case ftImaginary:
					while(first_point != last_point)
					{
						// Set the value in the output array
						const cv::Vec2f& element = coupled_images[coupled_image_index].at<cv::Vec2f>(*first_point++);
						assert(!std::isnan(element[1]));
						*dest++ = element[1];
					}
				break;
				case ftMagnitude:
					assert(false);
				break;
			}
			return;
		}
		// else do not do full image couple, wait and enter the following loop
		// for element-wise couping
	}

	// Loop through all the required points
	while(first_point != last_point)
	{
		std::complex<float> f1_val_cmplx;

		const cv::Point p = *first_point++;

		// Get the first raw feature
		if(f1_image_valid)
		{
			// Get the relevant element of the raw feature image
			const cv::Vec2f& element = raw_feats_unpadded[f1].at<cv::Vec2f>(p);
			f1_val_cmplx = std::complex<float>(element[0],element[1]);
		}
		else
			f1_val_cmplx = singleWindowFeature(f1,p);

		if(f2 == C_SECOND_FEATURE_NONE)
		{
			*dest++ = derivedFeatureFromComplex(f1_val_cmplx, type);
		}
		else
		{
			std::complex<float> f2_val_cmplx;

			// Get the second raw feature
			if(f2_image_valid)
			{
				const cv::Vec2f& element = raw_feats_unpadded[f2].at<cv::Vec2f>(p);
				f2_val_cmplx = std::complex<float>(element[0],element[1]);
			}
			else
				f2_val_cmplx = singleWindowFeature(f2,p);

			std::complex<float> coupled = coupleFeatures(f1_val_cmplx,raw_feat_r_list[f1],f2_val_cmplx,raw_feat_r_list[f2]);
			*dest++ = derivedFeatureFromComplex(coupled,type);
		}
	}
}

// Loop through precalculated images and extract the derived features from that
// require a single raw feature (i.e. no coupled). Must ensure that the relevent
// feature image is valid before calling this function!
template<typename TInputIterator,typename TOutputIterator>
void RIFeatExtractor::getDerivedFromSingleRawFeature(const int f, const featureType_enum type, TInputIterator first_point, const TInputIterator last_point, TOutputIterator dest)
{
	// Ensure the magnitude image is valid if required
	const int ind = (type == ftMagnitude) ?
		ensureMagnitudeImageValidity(f) :
		-1;

	switch(type)
	{
		case ftMagnitude:
			while(first_point != last_point)
			{
				// Find coordinates of the point in question
				*dest++ = raw_feature_magnitude_images[ind].at<float>(*first_point++);
			}
		break;
		case ftReal:
			while(first_point != last_point)
			{
				// Find coordinates of the point in question
				*dest++ = raw_feats_unpadded[f].at<cv::Vec2f>(*first_point++)[0];
			}
		break;
		case ftImaginary:
			while(first_point != last_point)
			{
				// Find coordinates of the point in question
				*dest++ = raw_feats_unpadded[f].at<cv::Vec2f>(*first_point++)[1];
			}
		break;
	}
}

/*! \brief Get a full feature vector for a single image location
*
* Takes a single image location and calculates the full feature vector of
* derived features at this image location in the input image. The input image
* must have been set up prior to calling this method. The resulting feature
* values are placed into the output container, which must be pre-allocated to
* the correct size (given by the \c getNumDerivedFeats() method).
*
* You must also ensure that all the points are valid, i.e. at least the
* 'halfsize' of the relevant basis function away from the image edge (see \c
* getMaxSpatBasisHalfsize()).
*
* This method is thread-safe in that it can be used in OpenMP parallel sections
* where other threads are calling this method or other thread-safe methods without
* causing data races.
*
* \tparam TOutputIterator The type of the iterator the output feature values. Must be at least an output iterator (according to C++ standard library definitions) and support assignment to a float value.
*
* \param point Image location, i.e. the centre of the detection window.
* \param dest Output iterator to the start of the output container where the derived feature scores will be placed. The container must have been pre-allocated to the correct size before calling.
*/
template<typename TOutputIterator>
void RIFeatExtractor::getDerivedFeatureVector(const cv::Point point, TOutputIterator dest)
{
	// Array to store raw feature values
	std::vector<std::complex<float>> raw_vector(num_raw_features);
	getRawFeatureVector(point, raw_vector.begin());

	std::complex<float> coupled;

	// Loop through derived features and calculate
	for(int d = 0; d < num_derived_features; ++d)
	{
		const int f1 = derived_feature_primary_list[d];
		const int f2 = derived_feature_secondary_list[d];
		const featureType_enum type = derived_feature_type_list[d];

		if(f2 == C_SECOND_FEATURE_NONE)
		{
			*dest++ = derivedFeatureFromComplex(raw_vector[f1],type);
		}
		else
		{
			// Check that the coupled feature is different from the last iteration
			if( (d == 0) || (f1 != derived_feature_primary_list[d-1]) || (f2 != derived_feature_secondary_list[d-1]) )
				coupled = coupleFeatures(raw_vector[f1],raw_feat_r_list[f1],raw_vector[f2],raw_feat_r_list[f2]);
			*dest++ = derivedFeatureFromComplex(coupled,type);
		}
	}

}

/*! \brief Get a full raw feature vector for a single image location
*
* Takes a single image location and calculates the full feature vector of
* raw features at this image location in the input image. The input image
* must have been set up prior to calling this method. The resulting feature
* values are placed into the output container, which must be pre-allocated to
* the correct size (given by the \c getNumRawFeats() method).
*
* You must also ensure that all the points are valid, i.e. at least the
* 'halfsize' of the relevant basis function away from the image edge (see \c
* getMaxSpatBasisHalfsize()).
*
* This method is thread-safe in that it can be used in OpenMP parallel sections
* where other threads are calling this method or other thread-safe methods without
* causing data races.
*
* \tparam TOutputIterator The type of the iterator the output feature values. Must be at least an output iterator (according to C++ standard library definitions) that derefences to a std::complex<float>.
*
* \param point Image location, i.e. the centre of the detection window.
* \param dest Output iterator to the start of the output container where the derived feature scores will be placed. The container must have been pre-allocated to the correct size before calling.
*/
template<typename TOutputIterator>
void RIFeatExtractor::getRawFeatureVector(const cv::Point point, TOutputIterator dest)
{
	// Calculate all the raw features
	for(int f = 0; f < num_raw_features; ++f)
	{
		if (checkRawFeatureValidity(f,always_use_frequency))
		{
			// Get the relevant element of the raw feature image
			const cv::Vec2f& element = raw_feats_unpadded[f].at<cv::Vec2f>(point);
			*dest++ = std::complex<float>(element[0],element[1]);
		}
		else
		{
			// Calculate using spatial filter
			*dest++ = singleWindowFeature(f,point);
		}
	}
}

/*! \brief Get the complex argument of a raw feature at a number of image locations
*
* Takes a sequence of image locations (each described by a cv::Point) and calculates
* the complex argument of the raw features at these image locations in the input image.
* The input image must have been set up prior to calling this method. The sine and cosine value of the resulting
* feature values are placed into the corresponding elements of the output sequences.
*
* You must also ensure that all the points are valid, i.e. at least the 'halfsize' of the relevant basis
* function away from the image edge (see \c getMaxSpatBasisHalfsize()).
* Note that these features are \b not rotation invariant.
*
* This method is thread-safe in that it can be used in OpenMP parallel sections
* where other threads are calling this method or other thread-safe methods without
* causing data races.
* \tparam TInputIterator The type of the iterator to the start of the input sequence of image locations. This must dereference to a cv::Point and be at least an input iterator (according to C++ standard library definitions).
* \tparam TOutputIterator The type of the iterator the output feature values. Must be at least an output iterator (according to C++ standard library definitions) and support assignment to a float value.
* \param first_point Iterator to the first image location, i.e. the centre of the first detection window.
* \param last_point Iterator to the end of the image location lists
* \param raw_feature_index The number of the raw feature whose argument should be calculated (must be between 0 (inclusive) and getNumRawFeats() (exclusive))
* \param cos_dest Output iterator to the start of the output sequence where the cosine of the raw feature argument will be placed.
* \param sin_dest Output iterator to the start of the output sequence where the sine of raw feature argument will be placed.
* \param flip_negative_rotation_orders If the raw feature has a negative effective rotation order, flip the output argument
* to create a feature with a positive effective rotation order. In effect this changes the sign on the sine output part.
*/
template<typename TInputIterator,typename TOutputIterator>
void RIFeatExtractor::getRawFeatureArg(TInputIterator first_point, TInputIterator last_point, const int raw_feature_index, TOutputIterator cos_dest, TOutputIterator sin_dest, const bool flip_negative_rotation_orders)
{
	const int num_points = std::distance(first_point,last_point);
	const bool calculate_if_invalid = always_use_frequency || (auto_method && ((raw_feat_usage_last_frame[raw_feature_index] > use_spatial_threshold || num_points > use_spatial_threshold)));
	const bool feat_image_valid = checkRawFeatureValidity(raw_feature_index,calculate_if_invalid);

	while(first_point != last_point)
	{
		std::complex<float> fval;

		if(feat_image_valid)
		{
			const cv::Vec2f& element = raw_feats_unpadded[raw_feature_index].at<cv::Vec2f>(*first_point++);
			fval = std::complex<float>(element[0],element[1]);
		}
		else
		{
			fval = singleWindowFeature(raw_feature_index,*first_point++);
		}

		// Calculate the sin and cos of the argument from the complex value
		const float mag = std::abs(fval);

		if(mag > 0.0)
		{
			*cos_dest++ = fval.real()/mag;
			// Flip if required by negating imaginary/sin part
			if(flip_negative_rotation_orders && (raw_feat_r_list[raw_feature_index] < 0) )
				*sin_dest++ = -fval.imag()/mag;
			else
				*sin_dest++ = fval.imag()/mag;
		}
		else
		{
			*cos_dest++ = 0.0;
			*sin_dest++ = 0.0;
		}

	}
}

} // end of namespace RIFeatures
