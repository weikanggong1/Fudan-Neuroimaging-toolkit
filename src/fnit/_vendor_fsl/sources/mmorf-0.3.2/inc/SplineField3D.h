/**
 *  \file
 *  \brief 3D basis-spline field implementing MMORF::WarpField
 *  \details Describes a warp in 3D where the (smooth) warp is parameterised in terms of uniformly spaced B-splines
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef SPLINE_FIELD_3D_H
#define SPLINE_FIELD_3D_H

// Local headers
#include <WarpField.h>
// STL headers
#include <vector>

/** Multi-Modal Registration Framework */
namespace MMORF
{
    /** 3D warp parameterised using B-splines */
    class SplineField3D : public WarpField
    {
        public:
            /** Field size and knot-spacing must be specified.
             *
             *  Non-integer knot-spacing is not permitted during construction */
            SplineField3D(std::vector<int> field_size, int knot_spacing);
            virtual ~SplineField3D();
            virtual void set_field_size(std::vector<int>);
            virtual std::vector<int> get_field_size() const;
            virtual std::vector<int> get_parameter_size() const;
            virtual std::vector<double> field() const;
            /** Current spline knot spacing.
             *
             *  This should match the knot spacing used to construct the spline field, 
             *  provided that the field has not been reparameterised.
             *  If the field is reparameterised by increasing the number of splines, then the
             *  knot spacing will decrease to facilitate this. */
            int get_knot_spacing() const;
            /** Reparameterise the field by upscaling the field dimensions.
             *
             *  Note that only integer scaling is allowed.
             *  Reparameterising preserves the shape of the current field exactly. */
            void scale_field(int scaling_factor);
            /** Reparameterise the field by upscaling the number of parameters.
             *
             *  Note that only integer scaling is allowed.
             *  Reparameterising preserves the shape of the current field exactly. */
            void scale_parameters(int scaling_factor);
        protected:
            /** Calculates the number of parameters necessary to fully describe a field.
             *
             *  Dependent on field size and knot-spacing */
            calculate_parameter_size();

            /** Spline knot-spacing.
             *
             *  May be non-integer if parameters have been scaled.
             *  Note that this may have the unwanted side-effect of splines no longer being
             *  centred on an integer point in the field. */
            int knot_spacing;
    }; // SplineField3D
} // MMORF
#endif // SPLINE_FIELD_3D_H
