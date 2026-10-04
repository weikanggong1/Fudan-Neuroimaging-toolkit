"""Strict-float32 eight-point sampler; NumPy/Numba dependencies only."""
from llvmlite import ir, binding
from numba import types
from numba.extending import intrinsic
from numba.core import cgutils


@intrinsic
def sample_block8(typing_context, data, values, weights, start,
                  base_x, base_y, base_z, step_x, step_y, step_z,
                  upper, smooth):
    arrays = (data, values, weights, upper, smooth)
    if not all(isinstance(array, types.Array) and array.dtype == types.float32 for array in arrays):
        return None
    if data.ndim != 3 or any(array.ndim != 1 for array in arrays[1:]):
        return None
    arguments = (data, values, weights, start, base_x, base_y, base_z,
                 step_x, step_y, step_z, upper, smooth)
    signature = types.UniTuple(types.boolean, 2)(*arguments)

    def codegen(context, builder, signature, arguments):
        data_value, output_values, output_weights, start_value = arguments[:4]
        data_array = context.make_array(signature.args[0])(context, builder, data_value)
        value_array = context.make_array(signature.args[1])(context, builder, output_values)
        weight_array = context.make_array(signature.args[2])(context, builder, output_weights)
        upper_array = context.make_array(signature.args[10])(context, builder, arguments[10])
        smooth_array = context.make_array(signature.args[11])(context, builder, arguments[11])
        float_type, integer_type, boolean_type = ir.FloatType(), ir.IntType(64), ir.IntType(1)
        floats = ir.VectorType(float_type, 8)
        integers = ir.VectorType(integer_type, 8)
        pointers = ir.VectorType(float_type.as_pointer(), 8)
        booleans = ir.VectorType(boolean_type, 8)
        zero, one = ir.Constant(floats, [0.0] * 8), ir.Constant(floats, [1.0] * 8)

        def splat(value, vector_type):
            result = ir.Constant(vector_type, ir.Undefined)
            for lane in range(8):
                result = builder.insert_element(result, value, ir.Constant(ir.IntType(32), lane))
            return result

        start_integer = context.cast(builder, start_value, signature.args[3], types.int64)
        indices = builder.add(splat(start_integer, integers), ir.Constant(integers, list(range(8))))
        positions = builder.sitofp(indices, floats)
        coordinates = [builder.fadd(splat(arguments[4 + axis], floats),
                                   builder.fmul(positions, splat(arguments[7 + axis], floats)))
                       for axis in range(3)]
        floor = cgutils.get_or_insert_function(builder.module, ir.FunctionType(floats, [floats]), 'llvm.floor.v8f32')
        axis_uppers = [splat(builder.load(builder.gep(upper_array.data, [ir.Constant(integer_type, axis)])), floats) for axis in range(3)]
        valid = ir.Constant(booleans, [True] * 8)
        interpolation_coordinates = []
        for coordinate, axis_upper in zip(coordinates, axis_uppers):
            valid = builder.and_(valid, builder.and_(builder.fcmp_ordered('>=', coordinate, zero),
                                                     builder.fcmp_ordered('<=', coordinate, axis_upper)))
            clamped = builder.select(builder.fcmp_ordered('<', coordinate, zero), zero, coordinate)
            clamped = builder.select(builder.fcmp_ordered('>', clamped, axis_upper), axis_upper, clamped)
            interpolation_coordinates.append(clamped)
        lower_float = [builder.call(floor, [coordinate]) for coordinate in interpolation_coordinates]
        lower = [builder.fptoui(value, integers) for value in lower_float]
        delta = [builder.fsub(interpolation_coordinates[axis], lower_float[axis]) for axis in range(3)]
        strides = cgutils.unpack_tuple(builder, data_array.strides, 3)
        stride_vectors = [splat(stride, integers) for stride in strides]
        data_bytes = builder.bitcast(data_array.data, ir.IntType(8).as_pointer())
        # LLVM <=15 uses typed pointer overload names; later versions use p0.
        pointer_overload = 'v8p0f32' if binding.llvm_version_info[0] <= 15 else 'v8p0'
        gather_type = ir.FunctionType(floats, [pointers, ir.IntType(32), booleans, floats])
        gather_function = cgutils.get_or_insert_function(builder.module, gather_type,
                                                         'llvm.masked.gather.v8f32.' + pointer_overload)
        full_mask = ir.Constant(booleans, [True] * 8)

        def gather(x, y, z):
            offset = builder.add(builder.add(builder.mul(x, stride_vectors[0]),
                                             builder.mul(y, stride_vectors[1])),
                                 builder.mul(z, stride_vectors[2]))
            pointer_vector = ir.Constant(pointers, ir.Undefined)
            for lane in range(8):
                index = ir.Constant(ir.IntType(32), lane)
                pointer = builder.gep(data_bytes, [builder.extract_element(offset, index)])
                pointer_vector = builder.insert_element(pointer_vector,
                                                        builder.bitcast(pointer, float_type.as_pointer()), index)
            return builder.call(gather_function, [pointer_vector, ir.Constant(ir.IntType(32), 4),
                                                  full_mask, ir.Constant(floats, ir.Undefined)])

        integer_one = ir.Constant(integers, [1] * 8)
        ix, iy, iz = lower
        ix1, iy1, iz1 = [builder.add(value, integer_one) for value in lower]
        d000, d100 = gather(ix, iy, iz), gather(ix1, iy, iz)
        d001, d101 = gather(ix, iy, iz1), gather(ix1, iy, iz1)
        d010, d110 = gather(ix, iy1, iz), gather(ix1, iy1, iz)
        d011, d111 = gather(ix, iy1, iz1), gather(ix1, iy1, iz1)
        dx, dy, dz = delta

        def interpolate(left, right, fraction):
            # Deliberately separate subtraction, multiplication and addition;
            # no LLVM fast-math flag or contraction is attached.
            return builder.fadd(builder.fmul(builder.fsub(right, left), fraction), left)

        temp1 = interpolate(d000, d100, dx)
        temp2 = interpolate(d001, d101, dx)
        temp3 = interpolate(d010, d110, dx)
        temp4 = interpolate(d011, d111, dx)
        sampled = interpolate(interpolate(temp1, temp3, dy), interpolate(temp2, temp4, dy), dz)
        taper_values = []
        for axis in range(3):
            scalar_upper = builder.load(builder.gep(upper_array.data, [ir.Constant(integer_type, axis)]))
            scalar_smooth = builder.load(builder.gep(smooth_array.data, [ir.Constant(integer_type, axis)]))
            axis_upper, axis_smooth = splat(scalar_upper, floats), splat(scalar_smooth, floats)
            coordinate = coordinates[axis]
            far = builder.fsub(axis_upper, coordinate)
            taper_values.append(builder.select(builder.fcmp_ordered('<', coordinate, axis_smooth),
                                                builder.fdiv(coordinate, axis_smooth),
                                                builder.select(builder.fcmp_ordered('<', far, axis_smooth),
                                                               builder.fdiv(far, axis_smooth), one)))
        geometric_weight = builder.fmul(builder.fmul(taper_values[0], taper_values[1]), taper_values[2])
        geometric_weight = builder.select(builder.fcmp_ordered('<', geometric_weight, zero), zero, geometric_weight)
        geometric_weight = builder.fmul(geometric_weight, builder.uitofp(valid, floats))
        sampled_finite = builder.fcmp_ordered('==', builder.fsub(sampled, sampled), zero)
        weight_finite = builder.fcmp_ordered('==', builder.fsub(geometric_weight, geometric_weight), zero)
        finite_mask = builder.and_(sampled_finite, weight_finite)
        all_finite = ir.Constant(boolean_type, True)
        any_valid = ir.Constant(boolean_type, False)
        for lane in range(8):
            all_finite = builder.and_(all_finite, builder.extract_element(finite_mask, ir.Constant(ir.IntType(32), lane)))
            any_valid = builder.or_(any_valid, builder.extract_element(valid, ir.Constant(ir.IntType(32), lane)))
        value_pointer = builder.bitcast(builder.gep(value_array.data, [start_integer]), floats.as_pointer())
        weight_pointer = builder.bitcast(builder.gep(weight_array.data, [start_integer]), floats.as_pointer())
        builder.store(sampled, value_pointer, align=4)
        builder.store(geometric_weight, weight_pointer, align=4)
        return context.make_tuple(builder, signature.return_type, (all_finite, any_valid))

    return signature, codegen


# Kept separate so the original unweighted sampler retains its LLVM operation
# sequence. Both kernels use float32 operations without contraction.
@intrinsic
def sample_block8_weighted(typing_context, data, values, weights, start,
                  base_x, base_y, base_z, step_x, step_y, step_z,
                  upper, smooth, moving_weight, reference_weight, reference_start):
    arrays = (data, values, weights, upper, smooth, moving_weight, reference_weight)
    if not all(isinstance(array, types.Array) and array.dtype == types.float32 for array in arrays):
        return None
    if data.ndim != 3 or moving_weight.ndim != 3 or any(array.ndim != 1 for array in (values, weights, upper, smooth, reference_weight)):
        return None
    arguments = (data, values, weights, start, base_x, base_y, base_z,
                 step_x, step_y, step_z, upper, smooth, moving_weight, reference_weight, reference_start)
    signature = types.UniTuple(types.boolean, 2)(*arguments)

    def codegen(context, builder, signature, arguments):
        data_value, output_values, output_weights, start_value = arguments[:4]
        data_array = context.make_array(signature.args[0])(context, builder, data_value)
        value_array = context.make_array(signature.args[1])(context, builder, output_values)
        weight_array = context.make_array(signature.args[2])(context, builder, output_weights)
        upper_array = context.make_array(signature.args[10])(context, builder, arguments[10])
        smooth_array = context.make_array(signature.args[11])(context, builder, arguments[11])
        moving_weight_array = context.make_array(signature.args[12])(context, builder, arguments[12])
        reference_weight_array = context.make_array(signature.args[13])(context, builder, arguments[13])
        reference_integer = context.cast(builder, arguments[14], signature.args[14], types.int64)
        float_type, integer_type, boolean_type = ir.FloatType(), ir.IntType(64), ir.IntType(1)
        floats = ir.VectorType(float_type, 8)
        integers = ir.VectorType(integer_type, 8)
        pointers = ir.VectorType(float_type.as_pointer(), 8)
        booleans = ir.VectorType(boolean_type, 8)
        zero, one = ir.Constant(floats, [0.0] * 8), ir.Constant(floats, [1.0] * 8)

        def splat(value, vector_type):
            result = ir.Constant(vector_type, ir.Undefined)
            for lane in range(8):
                result = builder.insert_element(result, value, ir.Constant(ir.IntType(32), lane))
            return result

        start_integer = context.cast(builder, start_value, signature.args[3], types.int64)
        indices = builder.add(splat(start_integer, integers), ir.Constant(integers, list(range(8))))
        positions = builder.sitofp(indices, floats)
        coordinates = [builder.fadd(splat(arguments[4 + axis], floats),
                                   builder.fmul(positions, splat(arguments[7 + axis], floats)))
                       for axis in range(3)]
        floor = cgutils.get_or_insert_function(builder.module, ir.FunctionType(floats, [floats]), 'llvm.floor.v8f32')
        axis_uppers = [splat(builder.load(builder.gep(upper_array.data, [ir.Constant(integer_type, axis)])), floats) for axis in range(3)]
        valid = ir.Constant(booleans, [True] * 8)
        interpolation_coordinates = []
        for coordinate, axis_upper in zip(coordinates, axis_uppers):
            valid = builder.and_(valid, builder.and_(builder.fcmp_ordered('>=', coordinate, zero),
                                                     builder.fcmp_ordered('<=', coordinate, axis_upper)))
            clamped = builder.select(builder.fcmp_ordered('<', coordinate, zero), zero, coordinate)
            clamped = builder.select(builder.fcmp_ordered('>', clamped, axis_upper), axis_upper, clamped)
            interpolation_coordinates.append(clamped)
        lower_float = [builder.call(floor, [coordinate]) for coordinate in interpolation_coordinates]
        lower = [builder.fptoui(value, integers) for value in lower_float]
        delta = [builder.fsub(interpolation_coordinates[axis], lower_float[axis]) for axis in range(3)]
        # LLVM <=15 uses typed pointer overload names; later versions use p0.
        pointer_overload = 'v8p0f32' if binding.llvm_version_info[0] <= 15 else 'v8p0'
        gather_type = ir.FunctionType(floats, [pointers, ir.IntType(32), booleans, floats])
        gather_function = cgutils.get_or_insert_function(builder.module, gather_type,
                                                         'llvm.masked.gather.v8f32.' + pointer_overload)
        full_mask = ir.Constant(booleans, [True] * 8)

        def load_offsets(array, offset):
            data_bytes = builder.bitcast(array.data, ir.IntType(8).as_pointer())
            pointer_vector = ir.Constant(pointers, ir.Undefined)
            for lane in range(8):
                index = ir.Constant(ir.IntType(32), lane)
                pointer = builder.gep(data_bytes, [builder.extract_element(offset, index)])
                pointer_vector = builder.insert_element(pointer_vector,
                                                        builder.bitcast(pointer, float_type.as_pointer()), index)
            return builder.call(gather_function, [pointer_vector, ir.Constant(ir.IntType(32), 4),
                                                  full_mask, ir.Constant(floats, ir.Undefined)])

        def gather(x, y, z, array):
            strides = cgutils.unpack_tuple(builder, array.strides, 3)
            stride_vectors = [splat(stride, integers) for stride in strides]
            offset = builder.add(builder.add(builder.mul(x, stride_vectors[0]),
                                             builder.mul(y, stride_vectors[1])),
                                 builder.mul(z, stride_vectors[2]))
            return load_offsets(array, offset)

        integer_one = ir.Constant(integers, [1] * 8)
        ix, iy, iz = lower
        ix1, iy1, iz1 = [builder.add(value, integer_one) for value in lower]
        d000, d100 = gather(ix, iy, iz, data_array), gather(ix1, iy, iz, data_array)
        d001, d101 = gather(ix, iy, iz1, data_array), gather(ix1, iy, iz1, data_array)
        d010, d110 = gather(ix, iy1, iz, data_array), gather(ix1, iy1, iz, data_array)
        d011, d111 = gather(ix, iy1, iz1, data_array), gather(ix1, iy1, iz1, data_array)
        dx, dy, dz = delta

        def interpolate(left, right, fraction):
            # Deliberately separate subtraction, multiplication and addition;
            # no LLVM fast-math flag or contraction is attached.
            return builder.fadd(builder.fmul(builder.fsub(right, left), fraction), left)

        temp1 = interpolate(d000, d100, dx)
        temp2 = interpolate(d001, d101, dx)
        temp3 = interpolate(d010, d110, dx)
        temp4 = interpolate(d011, d111, dx)
        sampled = interpolate(interpolate(temp1, temp3, dy), interpolate(temp2, temp4, dy), dz)
        taper_values = []
        for axis in range(3):
            scalar_upper = builder.load(builder.gep(upper_array.data, [ir.Constant(integer_type, axis)]))
            scalar_smooth = builder.load(builder.gep(smooth_array.data, [ir.Constant(integer_type, axis)]))
            axis_upper, axis_smooth = splat(scalar_upper, floats), splat(scalar_smooth, floats)
            coordinate = coordinates[axis]
            far = builder.fsub(axis_upper, coordinate)
            taper_values.append(builder.select(builder.fcmp_ordered('<', coordinate, axis_smooth),
                                                builder.fdiv(coordinate, axis_smooth),
                                                builder.select(builder.fcmp_ordered('<', far, axis_smooth),
                                                               builder.fdiv(far, axis_smooth), one)))
        geometric_weight = builder.fmul(builder.fmul(taper_values[0], taper_values[1]), taper_values[2])
        geometric_weight = builder.select(builder.fcmp_ordered('<', geometric_weight, zero), zero, geometric_weight)
        w000, w100 = gather(ix, iy, iz, moving_weight_array), gather(ix1, iy, iz, moving_weight_array)
        w001, w101 = gather(ix, iy, iz1, moving_weight_array), gather(ix1, iy, iz1, moving_weight_array)
        w010, w110 = gather(ix, iy1, iz, moving_weight_array), gather(ix1, iy1, iz, moving_weight_array)
        w011, w111 = gather(ix, iy1, iz1, moving_weight_array), gather(ix1, iy1, iz1, moving_weight_array)
        wt1, wt2 = interpolate(w000, w100, dx), interpolate(w001, w101, dx)
        wt3, wt4 = interpolate(w010, w110, dx), interpolate(w011, w111, dx)
        sampled_weight = interpolate(interpolate(wt1, wt3, dy), interpolate(wt2, wt4, dy), dz)
        reference_indices = builder.add(splat(reference_integer, integers), ir.Constant(integers, list(range(8))))
        reference_stride = cgutils.unpack_tuple(builder, reference_weight_array.strides, 1)[0]
        reference_values = load_offsets(reference_weight_array, builder.mul(reference_indices, splat(reference_stride, integers)))
        geometric_weight = builder.fmul(geometric_weight, sampled_weight)
        geometric_weight = builder.fmul(geometric_weight, reference_values)
        geometric_weight = builder.select(builder.fcmp_ordered('<', geometric_weight, zero), zero, geometric_weight)
        geometric_weight = builder.fmul(geometric_weight, builder.uitofp(valid, floats))
        sampled_finite = builder.fcmp_ordered('==', builder.fsub(sampled, sampled), zero)
        weight_finite = builder.fcmp_ordered('==', builder.fsub(geometric_weight, geometric_weight), zero)
        finite_mask = builder.and_(sampled_finite, weight_finite)
        all_finite = ir.Constant(boolean_type, True)
        any_valid = ir.Constant(boolean_type, False)
        for lane in range(8):
            all_finite = builder.and_(all_finite, builder.extract_element(finite_mask, ir.Constant(ir.IntType(32), lane)))
            any_valid = builder.or_(any_valid, builder.extract_element(valid, ir.Constant(ir.IntType(32), lane)))
        value_pointer = builder.bitcast(builder.gep(value_array.data, [start_integer]), floats.as_pointer())
        weight_pointer = builder.bitcast(builder.gep(weight_array.data, [start_integer]), floats.as_pointer())
        builder.store(sampled, value_pointer, align=4)
        builder.store(geometric_weight, weight_pointer, align=4)
        return context.make_tuple(builder, signature.return_type, (all_finite, any_valid))

    return signature, codegen
