"""CPU-only strict-FP64 eight-voxel spatial normal products.

Contiguous physical views and complete eight-voxel bounds must be proved
by the caller. False means no
output was stored: use the existing scalar expression for that whole block.
"""
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic


@intrinsic
def normal_block8(typing_context, field, weights, cross, output,
                  source, voxel, inner, scale, fit_scale, count):
    if not isinstance(weights, types.UniTuple) or weights.count != 9:
        return None
    if not isinstance(cross, types.UniTuple) or cross.count != 3:
        return None
    arrays = (field, weights.dtype, cross.dtype, output)
    if not all(isinstance(value, types.Array) and value.dtype == types.float64
               and value.ndim == 1 and value.layout == 'C' for value in arrays):
        return None
    if scale != types.float64 or count != types.float64 or fit_scale != types.boolean:
        return None
    arguments = (field, weights, cross, output, source, voxel, inner, scale, fit_scale, count)
    signature = types.boolean(*arguments)

    def codegen(context, builder, signature, arguments):
        dtype = ir.DoubleType()
        vector = ir.VectorType(dtype, 8)
        integer = ir.IntType(64)
        integer_vector = ir.VectorType(integer, 8)
        booleans = ir.VectorType(ir.IntType(1), 8)
        zero = ir.Constant(vector, [0.0] * 8)
        source = context.cast(builder, arguments[4], signature.args[4], types.int64)
        voxel = context.cast(builder, arguments[5], signature.args[5], types.int64)
        inner = context.cast(builder, arguments[6], signature.args[6], types.int64)
        field = context.make_array(signature.args[0])(context, builder, arguments[0])
        output = context.make_array(signature.args[3])(context, builder, arguments[3])
        weights = [context.make_array(signature.args[1].dtype)(context, builder, value)
                   for value in cgutils.unpack_tuple(builder, arguments[1], 9)]
        cross = [context.make_array(signature.args[2].dtype)(context, builder, value)
                 for value in cgutils.unpack_tuple(builder, arguments[2], 3)]

        def load(array, offset):
            pointer = builder.bitcast(builder.gep(array.data, [offset]), vector.as_pointer())
            return builder.load(pointer, align=8)

        def splat(value):
            result = ir.Constant(vector, ir.Undefined)
            for lane in range(8):
                result = builder.insert_element(result, value, ir.Constant(ir.IntType(32), lane))
            return result

        def finite(value):
            bits = builder.bitcast(value, integer_vector)
            exponent = builder.and_(bits, ir.Constant(integer_vector, [0x7ff0000000000000] * 8))
            return builder.icmp_unsigned('!=', exponent, ir.Constant(integer_vector, [0x7ff0000000000000] * 8))

        # All channel and weight loads precede any store, including aliasing
        # input/output views. Offsets describe physical channel groups.
        sources = [builder.add(source, builder.mul(inner, ir.Constant(integer, axis))) for axis in range(3)]
        d = [load(field, index) for index in sources]
        w = [load(array, voxel) for array in weights]
        c = [load(array, voxel) for array in cross]
        valid = ir.Constant(booleans, [True] * 8)
        for value in d + w:
            valid = builder.and_(valid, finite(value))
        cross_valid = ir.Constant(booleans, [True] * 8)
        for value in c:
            cross_valid = builder.and_(cross_valid, finite(value))
        valid = builder.and_(valid, builder.select(arguments[8], cross_valid,
                                                   ir.Constant(booleans, [True] * 8)))
        scalar_zero = ir.Constant(dtype, 0.0)
        scalar_valid = builder.fcmp_ordered('==', builder.fsub(arguments[9], arguments[9]), scalar_zero)
        scale_valid = builder.fcmp_ordered('==', builder.fsub(arguments[7], arguments[7]), scalar_zero)
        scalar_valid = builder.and_(scalar_valid,
            builder.select(arguments[8], scale_valid, ir.Constant(ir.IntType(1), True)))
        for lane in range(8):
            scalar_valid = builder.and_(scalar_valid,
                builder.extract_element(valid, ir.Constant(ir.IntType(32), lane)))

        compute_block = builder.append_basic_block('normal8_finite')
        fallback_block = builder.append_basic_block('normal8_scalar_fallback')
        merge_block = builder.append_basic_block('normal8_done')
        builder.cbranch(scalar_valid, compute_block, fallback_block)
        builder.position_at_end(compute_block)
        values = []
        for row in range(3):
            value = builder.fadd(zero, builder.fmul(w[3 * row], d[0]))
            value = builder.fadd(value, builder.fmul(w[3 * row + 1], d[1]))
            value = builder.fadd(value, builder.fmul(w[3 * row + 2], d[2]))
            values.append(value)

        scale_block = builder.append_basic_block('normal8_scale')
        no_scale_block = builder.append_basic_block('normal8_no_scale')
        divide_block = builder.append_basic_block('normal8_divide')
        builder.cbranch(arguments[8], scale_block, no_scale_block)
        builder.position_at_end(scale_block)
        scaled = [builder.fadd(value, builder.fmul(weight, splat(arguments[7])))
                  for value, weight in zip(values, c)]
        scale_predecessor = builder.block
        builder.branch(divide_block)
        builder.position_at_end(no_scale_block)
        no_scale_predecessor = builder.block
        builder.branch(divide_block)
        builder.position_at_end(divide_block)
        selections = []
        for value, scaled_value in zip(values, scaled):
            selected = builder.phi(vector)
            selected.add_incoming(scaled_value, scale_predecessor)
            selected.add_incoming(value, no_scale_predecessor)
            selections.append(selected)
        final = [builder.fdiv(selected, splat(arguments[9])) for selected in selections]
        for index, value in zip(sources, final):
            pointer = builder.bitcast(builder.gep(output.data, [index]), vector.as_pointer())
            builder.store(value, pointer, align=8)
        vector_predecessor = builder.block
        builder.branch(merge_block)

        builder.position_at_end(fallback_block)
        fallback_predecessor = builder.block
        builder.branch(merge_block)
        builder.position_at_end(merge_block)
        result = builder.phi(ir.IntType(1))
        result.add_incoming(ir.Constant(ir.IntType(1), True), vector_predecessor)
        result.add_incoming(ir.Constant(ir.IntType(1), False), fallback_predecessor)
        return result

    return signature, codegen
