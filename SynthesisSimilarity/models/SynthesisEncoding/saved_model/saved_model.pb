
ч
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28А
К
1material_encoder_11/zero_shift_11/zero_shift_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31material_encoder_11/zero_shift_11/zero_shift_bias
Г
Ematerial_encoder_11/zero_shift_11/zero_shift_bias/Read/ReadVariableOpReadVariableOp1material_encoder_11/zero_shift_11/zero_shift_bias*
_output_shapes
:*
dtype0
Ђ
#material_encoder_11/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S *4
shared_name%#material_encoder_11/dense_33/kernel

7material_encoder_11/dense_33/kernel/Read/ReadVariableOpReadVariableOp#material_encoder_11/dense_33/kernel*
_output_shapes

:S *
dtype0

!material_encoder_11/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!material_encoder_11/dense_33/bias

5material_encoder_11/dense_33/bias/Read/ReadVariableOpReadVariableOp!material_encoder_11/dense_33/bias*
_output_shapes
: *
dtype0
Ђ
#material_encoder_11/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *4
shared_name%#material_encoder_11/dense_34/kernel

7material_encoder_11/dense_34/kernel/Read/ReadVariableOpReadVariableOp#material_encoder_11/dense_34/kernel*
_output_shapes

:  *
dtype0

!material_encoder_11/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!material_encoder_11/dense_34/bias

5material_encoder_11/dense_34/bias/Read/ReadVariableOpReadVariableOp!material_encoder_11/dense_34/bias*
_output_shapes
: *
dtype0
Ђ
#material_encoder_11/dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *4
shared_name%#material_encoder_11/dense_35/kernel

7material_encoder_11/dense_35/kernel/Read/ReadVariableOpReadVariableOp#material_encoder_11/dense_35/kernel*
_output_shapes

:  *
dtype0

!material_encoder_11/dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!material_encoder_11/dense_35/bias

5material_encoder_11/dense_35/bias/Read/ReadVariableOpReadVariableOp!material_encoder_11/dense_35/bias*
_output_shapes
: *
dtype0

NoOpNoOp
ф
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
L
all_eles
all_ions
mat_encoder
	keras_api

signatures
 
 

zero_shift_layer

dens_1
hidden_layers
	uni_vec

	variables
trainable_variables
regularization_losses
	keras_api
 
 
w
zero_shift_bias

shift_bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

0
1
R
	variables
trainable_variables
regularization_losses
	keras_api
1
0
1
2
3
 4
!5
"6
1
0
1
2
3
 4
!5
"6
 
­
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics

	variables
trainable_variables
regularization_losses

VARIABLE_VALUE1material_encoder_11/zero_shift_11/zero_shift_biasGmat_encoder/zero_shift_layer/zero_shift_bias/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
mk
VARIABLE_VALUE#material_encoder_11/dense_33/kernel4mat_encoder/dens_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE!material_encoder_11/dense_33/bias2mat_encoder/dens_1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
h

kernel
 bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

!kernel
"bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
 
 
 
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
ki
VARIABLE_VALUE#material_encoder_11/dense_34/kernel2mat_encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE!material_encoder_11/dense_34/bias2mat_encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE#material_encoder_11/dense_35/kernel2mat_encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE!material_encoder_11/dense_35/bias2mat_encoder/variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
	4
 
 
 
 
 
 
 
 
 
 
 
 
 

0
 1

0
 1
 
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
2	variables
3trainable_variables
4regularization_losses

!0
"1

!0
"1
 
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
6	variables
7trainable_variables
8regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
serving_default_xPlaceholder*'
_output_shapes
:џџџџџџџџџS*
dtype0*
shape:џџџџџџџџџS
Ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_x1material_encoder_11/zero_shift_11/zero_shift_bias#material_encoder_11/dense_33/kernel!material_encoder_11/dense_33/bias#material_encoder_11/dense_34/kernel!material_encoder_11/dense_34/bias#material_encoder_11/dense_35/kernel!material_encoder_11/dense_35/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_6790362
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
П
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameEmaterial_encoder_11/zero_shift_11/zero_shift_bias/Read/ReadVariableOp7material_encoder_11/dense_33/kernel/Read/ReadVariableOp5material_encoder_11/dense_33/bias/Read/ReadVariableOp7material_encoder_11/dense_34/kernel/Read/ReadVariableOp5material_encoder_11/dense_34/bias/Read/ReadVariableOp7material_encoder_11/dense_35/kernel/Read/ReadVariableOp5material_encoder_11/dense_35/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_6791114
Ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename1material_encoder_11/zero_shift_11/zero_shift_bias#material_encoder_11/dense_33/kernel!material_encoder_11/dense_33/bias#material_encoder_11/dense_34/kernel!material_encoder_11/dense_34/bias#material_encoder_11/dense_35/kernel!material_encoder_11/dense_35/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_6791145гн

і
E__inference_dense_35_layer_call_and_return_conditional_losses_6790497

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

і
E__inference_dense_33_layer_call_and_return_conditional_losses_6790439

inputs0
matmul_readvariableop_resource:S -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
ѕ
h
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6791006

inputs
identityJ
SquareSquareinputs*
T0*'
_output_shapes
:џџџџџџџџџ `
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџy
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(L
SqrtSqrtSum:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+X
addAddV2Sqrt:y:0add/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
truedivRealDivinputsadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
є
Љ
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790958

inputs)
mul_readvariableop_resource:
identityЂmul/ReadVariableOpL
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
EqualEqualinputsEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџSX
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџSj
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0b
mulMulCast:y:0mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџSO
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџSV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS[
NoOpNoOp^mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџS: 2(
mul/ReadVariableOpmul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs

б
 __inference__traced_save_6791114
file_prefixP
Lsavev2_material_encoder_11_zero_shift_11_zero_shift_bias_read_readvariableopB
>savev2_material_encoder_11_dense_33_kernel_read_readvariableop@
<savev2_material_encoder_11_dense_33_bias_read_readvariableopB
>savev2_material_encoder_11_dense_34_kernel_read_readvariableop@
<savev2_material_encoder_11_dense_34_bias_read_readvariableopB
>savev2_material_encoder_11_dense_35_kernel_read_readvariableop@
<savev2_material_encoder_11_dense_35_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ж
valueЌBЉBGmat_encoder/zero_shift_layer/zero_shift_bias/.ATTRIBUTES/VARIABLE_VALUEB4mat_encoder/dens_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/dens_1/bias/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B џ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Lsavev2_material_encoder_11_zero_shift_11_zero_shift_bias_read_readvariableop>savev2_material_encoder_11_dense_33_kernel_read_readvariableop<savev2_material_encoder_11_dense_33_bias_read_readvariableop>savev2_material_encoder_11_dense_34_kernel_read_readvariableop<savev2_material_encoder_11_dense_34_bias_read_readvariableop>savev2_material_encoder_11_dense_35_kernel_read_readvariableop<savev2_material_encoder_11_dense_35_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*M
_input_shapes<
:: ::S : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::$ 

_output_shapes

:S : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: 

і
E__inference_dense_33_layer_call_and_return_conditional_losses_6790990

inputs0
matmul_readvariableop_resource:S -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Щ

*__inference_dense_35_layer_call_fn_6791047

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_6790497o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Чв
Ь
__inference___call___6790341
xK
=material_encoder_11_zero_shift_11_mul_readvariableop_resource:M
;material_encoder_11_dense_33_matmul_readvariableop_resource:S J
<material_encoder_11_dense_33_biasadd_readvariableop_resource: M
;material_encoder_11_dense_34_matmul_readvariableop_resource:  J
<material_encoder_11_dense_34_biasadd_readvariableop_resource: M
;material_encoder_11_dense_35_matmul_readvariableop_resource:  J
<material_encoder_11_dense_35_biasadd_readvariableop_resource: 
identityЂ3material_encoder_11/dense_33/BiasAdd/ReadVariableOpЂ2material_encoder_11/dense_33/MatMul/ReadVariableOpЂ3material_encoder_11/dense_34/BiasAdd/ReadVariableOpЂ2material_encoder_11/dense_34/MatMul/ReadVariableOpЂ3material_encoder_11/dense_35/BiasAdd/ReadVariableOpЂ2material_encoder_11/dense_35/MatMul/ReadVariableOpЂ4material_encoder_11/zero_shift_11/mul/ReadVariableOpJ
material_encoder_11/ShapeShapex*
T0*
_output_shapes
:c
material_encoder_11/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
material_encoder_11/NotEqualNotEqualx'material_encoder_11/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџSt
)material_encoder_11/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
material_encoder_11/AnyAny material_encoder_11/NotEqual:z:02material_encoder_11/Any/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџW
&material_encoder_11/boolean_mask/ShapeShapex*
T0*
_output_shapes
:~
4material_encoder_11/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6material_encoder_11/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6material_encoder_11/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
.material_encoder_11/boolean_mask/strided_sliceStridedSlice/material_encoder_11/boolean_mask/Shape:output:0=material_encoder_11/boolean_mask/strided_slice/stack:output:0?material_encoder_11/boolean_mask/strided_slice/stack_1:output:0?material_encoder_11/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7material_encoder_11/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Щ
%material_encoder_11/boolean_mask/ProdProd7material_encoder_11/boolean_mask/strided_slice:output:0@material_encoder_11/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: Y
(material_encoder_11/boolean_mask/Shape_1Shapex*
T0*
_output_shapes
:
6material_encoder_11/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8material_encoder_11/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
8material_encoder_11/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
0material_encoder_11/boolean_mask/strided_slice_1StridedSlice1material_encoder_11/boolean_mask/Shape_1:output:0?material_encoder_11/boolean_mask/strided_slice_1/stack:output:0Amaterial_encoder_11/boolean_mask/strided_slice_1/stack_1:output:0Amaterial_encoder_11/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskY
(material_encoder_11/boolean_mask/Shape_2Shapex*
T0*
_output_shapes
:
6material_encoder_11/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
8material_encoder_11/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
8material_encoder_11/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
0material_encoder_11/boolean_mask/strided_slice_2StridedSlice1material_encoder_11/boolean_mask/Shape_2:output:0?material_encoder_11/boolean_mask/strided_slice_2/stack:output:0Amaterial_encoder_11/boolean_mask/strided_slice_2/stack_1:output:0Amaterial_encoder_11/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
0material_encoder_11/boolean_mask/concat/values_1Pack.material_encoder_11/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:n
,material_encoder_11/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ
'material_encoder_11/boolean_mask/concatConcatV29material_encoder_11/boolean_mask/strided_slice_1:output:09material_encoder_11/boolean_mask/concat/values_1:output:09material_encoder_11/boolean_mask/strided_slice_2:output:05material_encoder_11/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:
(material_encoder_11/boolean_mask/ReshapeReshapex0material_encoder_11/boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
0material_encoder_11/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџР
*material_encoder_11/boolean_mask/Reshape_1Reshape material_encoder_11/Any:output:09material_encoder_11/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџ
&material_encoder_11/boolean_mask/WhereWhere3material_encoder_11/boolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџЈ
(material_encoder_11/boolean_mask/SqueezeSqueeze.material_encoder_11/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
p
.material_encoder_11/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
)material_encoder_11/boolean_mask/GatherV2GatherV21material_encoder_11/boolean_mask/Reshape:output:01material_encoder_11/boolean_mask/Squeeze:output:07material_encoder_11/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџSn
)material_encoder_11/zero_shift_11/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
'material_encoder_11/zero_shift_11/EqualEqual2material_encoder_11/boolean_mask/GatherV2:output:02material_encoder_11/zero_shift_11/Equal/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
&material_encoder_11/zero_shift_11/CastCast+material_encoder_11/zero_shift_11/Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџSЎ
4material_encoder_11/zero_shift_11/mul/ReadVariableOpReadVariableOp=material_encoder_11_zero_shift_11_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ш
%material_encoder_11/zero_shift_11/mulMul*material_encoder_11/zero_shift_11/Cast:y:0<material_encoder_11/zero_shift_11/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџSП
%material_encoder_11/zero_shift_11/addAddV22material_encoder_11/boolean_mask/GatherV2:output:0)material_encoder_11/zero_shift_11/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџSЎ
2material_encoder_11/dense_33/MatMul/ReadVariableOpReadVariableOp;material_encoder_11_dense_33_matmul_readvariableop_resource*
_output_shapes

:S *
dtype0Ц
#material_encoder_11/dense_33/MatMulMatMul)material_encoder_11/zero_shift_11/add:z:0:material_encoder_11/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ќ
3material_encoder_11/dense_33/BiasAdd/ReadVariableOpReadVariableOp<material_encoder_11_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Э
$material_encoder_11/dense_33/BiasAddBiasAdd-material_encoder_11/dense_33/MatMul:product:0;material_encoder_11/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_33/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Е
 material_encoder_11/dense_33/PowPow-material_encoder_11/dense_33/BiasAdd:output:0+material_encoder_11/dense_33/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_33/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=Ќ
 material_encoder_11/dense_33/mulMul+material_encoder_11/dense_33/mul/x:output:0$material_encoder_11/dense_33/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ А
 material_encoder_11/dense_33/addAddV2-material_encoder_11/dense_33/BiasAdd:output:0$material_encoder_11/dense_33/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_33/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?А
"material_encoder_11/dense_33/mul_1Mul-material_encoder_11/dense_33/mul_1/x:output:0$material_encoder_11/dense_33/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
!material_encoder_11/dense_33/TanhTanh&material_encoder_11/dense_33/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_33/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
"material_encoder_11/dense_33/add_1AddV2-material_encoder_11/dense_33/add_1/x:output:0%material_encoder_11/dense_33/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_33/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?В
"material_encoder_11/dense_33/mul_2Mul-material_encoder_11/dense_33/mul_2/x:output:0&material_encoder_11/dense_33/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ В
"material_encoder_11/dense_33/mul_3Mul-material_encoder_11/dense_33/BiasAdd:output:0&material_encoder_11/dense_33/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Ў
2material_encoder_11/dense_34/MatMul/ReadVariableOpReadVariableOp;material_encoder_11_dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0У
#material_encoder_11/dense_34/MatMulMatMul&material_encoder_11/dense_33/mul_3:z:0:material_encoder_11/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ќ
3material_encoder_11/dense_34/BiasAdd/ReadVariableOpReadVariableOp<material_encoder_11_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Э
$material_encoder_11/dense_34/BiasAddBiasAdd-material_encoder_11/dense_34/MatMul:product:0;material_encoder_11/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_34/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Е
 material_encoder_11/dense_34/PowPow-material_encoder_11/dense_34/BiasAdd:output:0+material_encoder_11/dense_34/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_34/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=Ќ
 material_encoder_11/dense_34/mulMul+material_encoder_11/dense_34/mul/x:output:0$material_encoder_11/dense_34/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ А
 material_encoder_11/dense_34/addAddV2-material_encoder_11/dense_34/BiasAdd:output:0$material_encoder_11/dense_34/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_34/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?А
"material_encoder_11/dense_34/mul_1Mul-material_encoder_11/dense_34/mul_1/x:output:0$material_encoder_11/dense_34/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
!material_encoder_11/dense_34/TanhTanh&material_encoder_11/dense_34/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_34/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
"material_encoder_11/dense_34/add_1AddV2-material_encoder_11/dense_34/add_1/x:output:0%material_encoder_11/dense_34/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_34/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?В
"material_encoder_11/dense_34/mul_2Mul-material_encoder_11/dense_34/mul_2/x:output:0&material_encoder_11/dense_34/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ В
"material_encoder_11/dense_34/mul_3Mul-material_encoder_11/dense_34/BiasAdd:output:0&material_encoder_11/dense_34/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Ў
2material_encoder_11/dense_35/MatMul/ReadVariableOpReadVariableOp;material_encoder_11_dense_35_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0У
#material_encoder_11/dense_35/MatMulMatMul&material_encoder_11/dense_34/mul_3:z:0:material_encoder_11/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ќ
3material_encoder_11/dense_35/BiasAdd/ReadVariableOpReadVariableOp<material_encoder_11_dense_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Э
$material_encoder_11/dense_35/BiasAddBiasAdd-material_encoder_11/dense_35/MatMul:product:0;material_encoder_11/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_35/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Е
 material_encoder_11/dense_35/PowPow-material_encoder_11/dense_35/BiasAdd:output:0+material_encoder_11/dense_35/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
"material_encoder_11/dense_35/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=Ќ
 material_encoder_11/dense_35/mulMul+material_encoder_11/dense_35/mul/x:output:0$material_encoder_11/dense_35/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ А
 material_encoder_11/dense_35/addAddV2-material_encoder_11/dense_35/BiasAdd:output:0$material_encoder_11/dense_35/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_35/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?А
"material_encoder_11/dense_35/mul_1Mul-material_encoder_11/dense_35/mul_1/x:output:0$material_encoder_11/dense_35/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
!material_encoder_11/dense_35/TanhTanh&material_encoder_11/dense_35/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_35/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
"material_encoder_11/dense_35/add_1AddV2-material_encoder_11/dense_35/add_1/x:output:0%material_encoder_11/dense_35/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ i
$material_encoder_11/dense_35/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?В
"material_encoder_11/dense_35/mul_2Mul-material_encoder_11/dense_35/mul_2/x:output:0&material_encoder_11/dense_35/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ В
"material_encoder_11/dense_35/mul_3Mul-material_encoder_11/dense_35/BiasAdd:output:0&material_encoder_11/dense_35/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*material_encoder_11/unify_vector_14/SquareSquare&material_encoder_11/dense_35/mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
9material_encoder_11/unify_vector_14/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџх
'material_encoder_11/unify_vector_14/SumSum.material_encoder_11/unify_vector_14/Square:y:0Bmaterial_encoder_11/unify_vector_14/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(
(material_encoder_11/unify_vector_14/SqrtSqrt0material_encoder_11/unify_vector_14/Sum:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
)material_encoder_11/unify_vector_14/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+Ф
'material_encoder_11/unify_vector_14/addAddV2,material_encoder_11/unify_vector_14/Sqrt:y:02material_encoder_11/unify_vector_14/add/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџН
+material_encoder_11/unify_vector_14/truedivRealDiv&material_encoder_11/dense_35/mul_3:z:0+material_encoder_11/unify_vector_14/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ q
'material_encoder_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)material_encoder_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)material_encoder_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!material_encoder_11/strided_sliceStridedSlice"material_encoder_11/Shape:output:00material_encoder_11/strided_slice/stack:output:02material_encoder_11/strided_slice/stack_1:output:02material_encoder_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
material_encoder_11/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
material_encoder_11/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ч
material_encoder_11/rangeRange(material_encoder_11/range/start:output:0*material_encoder_11/strided_slice:output:0(material_encoder_11/range/delta:output:0*#
_output_shapes
:џџџџџџџџџz
(material_encoder_11/boolean_mask_1/ShapeShape"material_encoder_11/range:output:0*
T0*
_output_shapes
:
6material_encoder_11/boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8material_encoder_11/boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8material_encoder_11/boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
0material_encoder_11/boolean_mask_1/strided_sliceStridedSlice1material_encoder_11/boolean_mask_1/Shape:output:0?material_encoder_11/boolean_mask_1/strided_slice/stack:output:0Amaterial_encoder_11/boolean_mask_1/strided_slice/stack_1:output:0Amaterial_encoder_11/boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
9material_encoder_11/boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
'material_encoder_11/boolean_mask_1/ProdProd9material_encoder_11/boolean_mask_1/strided_slice:output:0Bmaterial_encoder_11/boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: |
*material_encoder_11/boolean_mask_1/Shape_1Shape"material_encoder_11/range:output:0*
T0*
_output_shapes
:
8material_encoder_11/boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:material_encoder_11/boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:material_encoder_11/boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2material_encoder_11/boolean_mask_1/strided_slice_1StridedSlice3material_encoder_11/boolean_mask_1/Shape_1:output:0Amaterial_encoder_11/boolean_mask_1/strided_slice_1/stack:output:0Cmaterial_encoder_11/boolean_mask_1/strided_slice_1/stack_1:output:0Cmaterial_encoder_11/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask|
*material_encoder_11/boolean_mask_1/Shape_2Shape"material_encoder_11/range:output:0*
T0*
_output_shapes
:
8material_encoder_11/boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
:material_encoder_11/boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:material_encoder_11/boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2material_encoder_11/boolean_mask_1/strided_slice_2StridedSlice3material_encoder_11/boolean_mask_1/Shape_2:output:0Amaterial_encoder_11/boolean_mask_1/strided_slice_2/stack:output:0Cmaterial_encoder_11/boolean_mask_1/strided_slice_2/stack_1:output:0Cmaterial_encoder_11/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
2material_encoder_11/boolean_mask_1/concat/values_1Pack0material_encoder_11/boolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:p
.material_encoder_11/boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : г
)material_encoder_11/boolean_mask_1/concatConcatV2;material_encoder_11/boolean_mask_1/strided_slice_1:output:0;material_encoder_11/boolean_mask_1/concat/values_1:output:0;material_encoder_11/boolean_mask_1/strided_slice_2:output:07material_encoder_11/boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
*material_encoder_11/boolean_mask_1/ReshapeReshape"material_encoder_11/range:output:02material_encoder_11/boolean_mask_1/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
2material_encoder_11/boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџФ
,material_encoder_11/boolean_mask_1/Reshape_1Reshape material_encoder_11/Any:output:0;material_encoder_11/boolean_mask_1/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџ
(material_encoder_11/boolean_mask_1/WhereWhere5material_encoder_11/boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџЌ
*material_encoder_11/boolean_mask_1/SqueezeSqueeze0material_encoder_11/boolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
r
0material_encoder_11/boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ­
+material_encoder_11/boolean_mask_1/GatherV2GatherV23material_encoder_11/boolean_mask_1/Reshape:output:03material_encoder_11/boolean_mask_1/Squeeze:output:09material_encoder_11/boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџs
)material_encoder_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+material_encoder_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+material_encoder_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
#material_encoder_11/strided_slice_1StridedSlice"material_encoder_11/Shape:output:02material_encoder_11/strided_slice_1/stack:output:04material_encoder_11/strided_slice_1/stack_1:output:04material_encoder_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
material_encoder_11/Shape_1Shape/material_encoder_11/unify_vector_14/truediv:z:0*
T0*
_output_shapes
:s
)material_encoder_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+material_encoder_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+material_encoder_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
#material_encoder_11/strided_slice_2StridedSlice$material_encoder_11/Shape_1:output:02material_encoder_11/strided_slice_2/stack:output:04material_encoder_11/strided_slice_2/stack_1:output:04material_encoder_11/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maska
material_encoder_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : к
material_encoder_11/concatConcatV2,material_encoder_11/strided_slice_1:output:0,material_encoder_11/strided_slice_2:output:0(material_encoder_11/concat/axis:output:0*
N*
T0*
_output_shapes
:d
"material_encoder_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :С
material_encoder_11/ExpandDims
ExpandDims4material_encoder_11/boolean_mask_1/GatherV2:output:0+material_encoder_11/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџы
material_encoder_11/ScatterNd	ScatterNd'material_encoder_11/ExpandDims:output:0/material_encoder_11/unify_vector_14/truediv:z:0#material_encoder_11/concat:output:0*
T0*
Tindices0*'
_output_shapes
:џџџџџџџџџ e
 material_encoder_11/NotEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
material_encoder_11/NotEqual_1NotEqualx)material_encoder_11/NotEqual_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџSv
+material_encoder_11/Any_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
material_encoder_11/Any_1Any"material_encoder_11/NotEqual_1:z:04material_encoder_11/Any_1/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџu
IdentityIdentity&material_encoder_11/ScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ О
NoOpNoOp4^material_encoder_11/dense_33/BiasAdd/ReadVariableOp3^material_encoder_11/dense_33/MatMul/ReadVariableOp4^material_encoder_11/dense_34/BiasAdd/ReadVariableOp3^material_encoder_11/dense_34/MatMul/ReadVariableOp4^material_encoder_11/dense_35/BiasAdd/ReadVariableOp3^material_encoder_11/dense_35/MatMul/ReadVariableOp5^material_encoder_11/zero_shift_11/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 2j
3material_encoder_11/dense_33/BiasAdd/ReadVariableOp3material_encoder_11/dense_33/BiasAdd/ReadVariableOp2h
2material_encoder_11/dense_33/MatMul/ReadVariableOp2material_encoder_11/dense_33/MatMul/ReadVariableOp2j
3material_encoder_11/dense_34/BiasAdd/ReadVariableOp3material_encoder_11/dense_34/BiasAdd/ReadVariableOp2h
2material_encoder_11/dense_34/MatMul/ReadVariableOp2material_encoder_11/dense_34/MatMul/ReadVariableOp2j
3material_encoder_11/dense_35/BiasAdd/ReadVariableOp3material_encoder_11/dense_35/BiasAdd/ReadVariableOp2h
2material_encoder_11/dense_35/MatMul/ReadVariableOp2material_encoder_11/dense_35/MatMul/ReadVariableOp2l
4material_encoder_11/zero_shift_11/mul/ReadVariableOp4material_encoder_11/zero_shift_11/mul/ReadVariableOp:J F
'
_output_shapes
:џџџџџџџџџS

_user_specified_namex
І

/__inference_zero_shift_11_layer_call_fn_6790947

inputs
unknown:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790412o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџS: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs

і
E__inference_dense_34_layer_call_and_return_conditional_losses_6790468

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ћ

P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790940

inputs7
)zero_shift_11_mul_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource:S 6
(dense_33_biasadd_readvariableop_resource: 9
'dense_34_matmul_readvariableop_resource:  6
(dense_34_biasadd_readvariableop_resource: 9
'dense_35_matmul_readvariableop_resource:  6
(dense_35_biasadd_readvariableop_resource: 
identity

identity_1

identity_2Ђdense_33/BiasAdd/ReadVariableOpЂdense_33/MatMul/ReadVariableOpЂdense_34/BiasAdd/ReadVariableOpЂdense_34/MatMul/ReadVariableOpЂdense_35/BiasAdd/ReadVariableOpЂdense_35/MatMul/ReadVariableOpЂ zero_shift_11/mul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    c
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ]
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџH
boolean_mask/ShapeShapeinputs*
T0*
_output_shapes
:j
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:m
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: J
boolean_mask/Shape_1Shapeinputs*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskJ
boolean_mask/Shape_2Shapeinputs*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:Z
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:w
boolean_mask/ReshapeReshapeinputsboolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџSo
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask/Reshape_1ReshapeAny:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџe
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
\
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : й
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџSZ
zero_shift_11/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
zero_shift_11/EqualEqualboolean_mask/GatherV2:output:0zero_shift_11/Equal/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџSt
zero_shift_11/CastCastzero_shift_11/Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџS
 zero_shift_11/mul/ReadVariableOpReadVariableOp)zero_shift_11_mul_readvariableop_resource*
_output_shapes
:*
dtype0
zero_shift_11/mulMulzero_shift_11/Cast:y:0(zero_shift_11/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS
zero_shift_11/addAddV2boolean_mask/GatherV2:output:0zero_shift_11/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџS
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:S *
dtype0
dense_33/MatMulMatMulzero_shift_11/add:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_33/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@y
dense_33/PowPowdense_33/BiasAdd:output:0dense_33/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_33/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=p
dense_33/mulMuldense_33/mul/x:output:0dense_33/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
dense_33/addAddV2dense_33/BiasAdd:output:0dense_33/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_33/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?t
dense_33/mul_1Muldense_33/mul_1/x:output:0dense_33/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
dense_33/TanhTanhdense_33/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_33/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
dense_33/add_1AddV2dense_33/add_1/x:output:0dense_33/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_33/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dense_33/mul_2Muldense_33/mul_2/x:output:0dense_33/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ v
dense_33/mul_3Muldense_33/BiasAdd:output:0dense_33/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_34/MatMulMatMuldense_33/mul_3:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_34/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@y
dense_34/PowPowdense_34/BiasAdd:output:0dense_34/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_34/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=p
dense_34/mulMuldense_34/mul/x:output:0dense_34/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
dense_34/addAddV2dense_34/BiasAdd:output:0dense_34/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_34/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?t
dense_34/mul_1Muldense_34/mul_1/x:output:0dense_34/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
dense_34/TanhTanhdense_34/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_34/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
dense_34/add_1AddV2dense_34/add_1/x:output:0dense_34/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_34/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dense_34/mul_2Muldense_34/mul_2/x:output:0dense_34/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ v
dense_34/mul_3Muldense_34/BiasAdd:output:0dense_34/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_35/MatMulMatMuldense_34/mul_3:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_35/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@y
dense_35/PowPowdense_35/BiasAdd:output:0dense_35/Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
dense_35/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=p
dense_35/mulMuldense_35/mul/x:output:0dense_35/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
dense_35/addAddV2dense_35/BiasAdd:output:0dense_35/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_35/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?t
dense_35/mul_1Muldense_35/mul_1/x:output:0dense_35/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
dense_35/TanhTanhdense_35/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_35/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
dense_35/add_1AddV2dense_35/add_1/x:output:0dense_35/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ U
dense_35/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dense_35/mul_2Muldense_35/mul_2/x:output:0dense_35/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ v
dense_35/mul_3Muldense_35/BiasAdd:output:0dense_35/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ f
unify_vector_14/SquareSquaredense_35/mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ p
%unify_vector_14/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЉ
unify_vector_14/SumSumunify_vector_14/Square:y:0.unify_vector_14/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(l
unify_vector_14/SqrtSqrtunify_vector_14/Sum:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
unify_vector_14/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+
unify_vector_14/addAddV2unify_vector_14/Sqrt:y:0unify_vector_14/add/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
unify_vector_14/truedivRealDivdense_35/mul_3:z:0unify_vector_14/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :w
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџR
boolean_mask_1/ShapeShaperange:output:0*
T0*
_output_shapes
:l
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:o
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: T
boolean_mask_1/Shape_1Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskT
boolean_mask_1/Shape_2Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskr
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:\
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : я
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
boolean_mask_1/ReshapeReshaperange:output:0boolean_mask_1/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџq
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask_1/Reshape_1ReshapeAny:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџi
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
^
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : н
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskR
Shape_1Shapeunify_vector_14/truediv:z:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice_1:output:0strided_slice_2:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDims boolean_mask_1/GatherV2:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
	ScatterNd	ScatterNdExpandDims:output:0unify_vector_14/truediv:z:0concat:output:0*
T0*
Tindices0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_2IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ В
NoOpNoOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp!^zero_shift_11/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2D
 zero_shift_11/mul/ReadVariableOp zero_shift_11/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
ѕ
h
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6790514

inputs
identityJ
SquareSquareinputs*
T0*'
_output_shapes
:џџџџџџџџџ `
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџy
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(L
SqrtSqrtSum:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+X
addAddV2Sqrt:y:0add/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
truedivRealDivinputsadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
є
Љ
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790412

inputs)
mul_readvariableop_resource:
identityЂmul/ReadVariableOpL
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
EqualEqualinputsEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџSX
CastCast	Equal:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџSj
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0b
mulMulCast:y:0mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџSO
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџSV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS[
NoOpNoOp^mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџS: 2(
mul/ReadVariableOpmul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Рm

P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790567

inputs#
zero_shift_11_6790413:"
dense_33_6790440:S 
dense_33_6790442: "
dense_34_6790469:  
dense_34_6790471: "
dense_35_6790498:  
dense_35_6790500: 
identity

identity_1

identity_2Ђ dense_33/StatefulPartitionedCallЂ dense_34/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂ%zero_shift_11/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    c
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ]
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџH
boolean_mask/ShapeShapeinputs*
T0*
_output_shapes
:j
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:m
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: J
boolean_mask/Shape_1Shapeinputs*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskJ
boolean_mask/Shape_2Shapeinputs*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:Z
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:w
boolean_mask/ReshapeReshapeinputsboolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџSo
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask/Reshape_1ReshapeAny:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџe
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
\
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : й
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџS
%zero_shift_11/StatefulPartitionedCallStatefulPartitionedCallboolean_mask/GatherV2:output:0zero_shift_11_6790413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790412 
 dense_33/StatefulPartitionedCallStatefulPartitionedCall.zero_shift_11/StatefulPartitionedCall:output:0dense_33_6790440dense_33_6790442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_6790439
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_6790469dense_34_6790471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_6790468
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_6790498dense_35_6790500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_6790497я
unify_vector_14/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6790514]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :w
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџR
boolean_mask_1/ShapeShaperange:output:0*
T0*
_output_shapes
:l
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:o
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: T
boolean_mask_1/Shape_1Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskT
boolean_mask_1/Shape_2Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskr
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:\
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : я
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
boolean_mask_1/ReshapeReshaperange:output:0boolean_mask_1/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџq
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask_1/Reshape_1ReshapeAny:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџi
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
^
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : н
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask_
Shape_1Shape(unify_vector_14/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice_1:output:0strided_slice_2:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDims boolean_mask_1/GatherV2:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
	ScatterNd	ScatterNdExpandDims:output:0(unify_vector_14/PartitionedCall:output:0concat:output:0*
T0*
Tindices0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_2IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ з
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall&^zero_shift_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%zero_shift_11/StatefulPartitionedCall%zero_shift_11/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs

і
E__inference_dense_34_layer_call_and_return_conditional_losses_6791038

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
І
Ш
5__inference_material_encoder_11_layer_call_fn_6790588
input_1
unknown:
	unknown_0:S 
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5: 
identity

identity_1

identity_2ЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџS
!
_user_specified_name	input_1
Ѓ
Ч
5__inference_material_encoder_11_layer_call_fn_6790783

inputs
unknown:
	unknown_0:S 
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5: 
identity

identity_1

identity_2ЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2 *0J 8 *Y
fTRR
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
ь"
ж
#__inference__traced_restore_6791145
file_prefixP
Bassignvariableop_material_encoder_11_zero_shift_11_zero_shift_bias:H
6assignvariableop_1_material_encoder_11_dense_33_kernel:S B
4assignvariableop_2_material_encoder_11_dense_33_bias: H
6assignvariableop_3_material_encoder_11_dense_34_kernel:  B
4assignvariableop_4_material_encoder_11_dense_34_bias: H
6assignvariableop_5_material_encoder_11_dense_35_kernel:  B
4assignvariableop_6_material_encoder_11_dense_35_bias: 

identity_8ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ж
valueЌBЉBGmat_encoder/zero_shift_layer/zero_shift_bias/.ATTRIBUTES/VARIABLE_VALUEB4mat_encoder/dens_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/dens_1/bias/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB2mat_encoder/variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Ц
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOpAssignVariableOpBassignvariableop_material_encoder_11_zero_shift_11_zero_shift_biasIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_1AssignVariableOp6assignvariableop_1_material_encoder_11_dense_33_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_2AssignVariableOp4assignvariableop_2_material_encoder_11_dense_33_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_3AssignVariableOp6assignvariableop_3_material_encoder_11_dense_34_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_4AssignVariableOp4assignvariableop_4_material_encoder_11_dense_34_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_5AssignVariableOp6assignvariableop_5_material_encoder_11_dense_35_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_6AssignVariableOp4assignvariableop_6_material_encoder_11_dense_35_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ы

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: й
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
В
M
1__inference_unify_vector_14_layer_call_fn_6790995

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6790514`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

і
E__inference_dense_35_layer_call_and_return_conditional_losses_6791070

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@^
PowPowBiasAdd:output:0Pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=U
mulMulmul/x:output:0Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
addAddV2BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?Y
mul_1Mulmul_1/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?\
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
mul_2Mulmul_2/x:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ [
mul_3MulBiasAdd:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentity	mul_3:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Шm

P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790760
input_1#
zero_shift_11_6790690:"
dense_33_6790693:S 
dense_33_6790695: "
dense_34_6790698:  
dense_34_6790700: "
dense_35_6790703:  
dense_35_6790705: 
identity

identity_1

identity_2Ђ dense_33/StatefulPartitionedCallЂ dense_34/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂ%zero_shift_11/StatefulPartitionedCall<
ShapeShapeinput_1*
T0*
_output_shapes
:O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    d
NotEqualNotEqualinput_1NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ]
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџI
boolean_mask/ShapeShapeinput_1*
T0*
_output_shapes
:j
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:m
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: K
boolean_mask/Shape_1Shapeinput_1*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskK
boolean_mask/Shape_2Shapeinput_1*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:Z
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:x
boolean_mask/ReshapeReshapeinput_1boolean_mask/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџSo
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask/Reshape_1ReshapeAny:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџe
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
\
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : й
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџS
%zero_shift_11/StatefulPartitionedCallStatefulPartitionedCallboolean_mask/GatherV2:output:0zero_shift_11_6790690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790412 
 dense_33/StatefulPartitionedCallStatefulPartitionedCall.zero_shift_11/StatefulPartitionedCall:output:0dense_33_6790693dense_33_6790695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_6790439
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_6790698dense_34_6790700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_6790468
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_6790703dense_35_6790705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_6790497я
unify_vector_14/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6790514]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :w
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџR
boolean_mask_1/ShapeShaperange:output:0*
T0*
_output_shapes
:l
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:o
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: T
boolean_mask_1/Shape_1Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskT
boolean_mask_1/Shape_2Shaperange:output:0*
T0*
_output_shapes
:n
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskr
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:\
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : я
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
boolean_mask_1/ReshapeReshaperange:output:0boolean_mask_1/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџq
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask_1/Reshape_1ReshapeAny:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџi
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
^
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : н
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask_
Shape_1Shape(unify_vector_14/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice_1:output:0strided_slice_2:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDims boolean_mask_1/GatherV2:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
	ScatterNd	ScatterNdExpandDims:output:0(unify_vector_14/PartitionedCall:output:0concat:output:0*
T0*
Tindices0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_2IdentityScatterNd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ з
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall&^zero_shift_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%zero_shift_11/StatefulPartitionedCall%zero_shift_11/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџS
!
_user_specified_name	input_1
Щ

*__inference_dense_34_layer_call_fn_6791015

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_6790468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Щ

*__inference_dense_33_layer_call_fn_6790967

inputs
unknown:S 
	unknown_0: 
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_6790439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
м

%__inference_signature_wrapper_6790362
x
unknown:
	unknown_0:S 
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5: 
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *)
_read_only_resource_inputs
	*2
config_proto" 

CPU

GPU2 *0J 8 *%
f R
__inference___call___6790341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџS: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:џџџџџџџџџS

_user_specified_namex"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
/
x*
serving_default_x:0џџџџџџџџџS<
output_00
StatefulPartitionedCall:0џџџџџџџџџ tensorflow/serving/predict:ђZ
s
all_eles
all_ions
mat_encoder
	keras_api

signatures
I__call__"
_tf_keras_model
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ч
zero_shift_layer

dens_1
hidden_layers
	uni_vec

	variables
trainable_variables
regularization_losses
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_model
"
_generic_user_object
,
Lserving_default"
signature_map
Ъ
zero_shift_bias

shift_bias
	variables
trainable_variables
regularization_losses
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
 "
trackable_list_wrapper
­
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics

	variables
trainable_variables
regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
?:=21material_encoder_11/zero_shift_11/zero_shift_bias
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
5:3S 2#material_encoder_11/dense_33/kernel
/:- 2!material_encoder_11/dense_33/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Л

kernel
 bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

!kernel
"bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
5:3  2#material_encoder_11/dense_34/kernel
/:- 2!material_encoder_11/dense_34/bias
5:3  2#material_encoder_11/dense_35/kernel
/:- 2!material_encoder_11/dense_35/bias
 "
trackable_list_wrapper
C
0
1
2
3
	4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
2	variables
3trainable_variables
4regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
6	variables
7trainable_variables
8regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
м2й
__inference___call___6790341И
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
џџџџџџџџџS
Ѓ2 
5__inference_material_encoder_11_layer_call_fn_6790588
5__inference_material_encoder_11_layer_call_fn_6790783Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790940
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790760Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЦBУ
%__inference_signature_wrapper_6790362x"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_zero_shift_11_layer_call_fn_6790947Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790958Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_33_layer_call_fn_6790967Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_33_layer_call_and_return_conditional_losses_6790990Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_unify_vector_14_layer_call_fn_6790995Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6791006Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_34_layer_call_fn_6791015Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_34_layer_call_and_return_conditional_losses_6791038Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_35_layer_call_fn_6791047Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_35_layer_call_and_return_conditional_losses_6791070Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 o
__inference___call___6790341O !"*Ђ'
 Ђ

xџџџџџџџџџS
Њ "џџџџџџџџџ Ѕ
E__inference_dense_33_layer_call_and_return_conditional_losses_6790990\/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_33_layer_call_fn_6790967O/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "џџџџџџџџџ Ѕ
E__inference_dense_34_layer_call_and_return_conditional_losses_6791038\ /Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_34_layer_call_fn_6791015O /Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ѕ
E__inference_dense_35_layer_call_and_return_conditional_losses_6791070\!"/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_35_layer_call_fn_6791047O!"/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ 
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790760Ћ !"4Ђ1
*Ђ'
!
input_1џџџџџџџџџS

 
Њ "jЂg
`Ђ]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 џ
P__inference_material_encoder_11_layer_call_and_return_conditional_losses_6790940Њ !"3Ђ0
)Ђ&
 
inputsџџџџџџџџџS

 
Њ "jЂg
`Ђ]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 е
5__inference_material_encoder_11_layer_call_fn_6790588 !"4Ђ1
*Ђ'
!
input_1џџџџџџџџџS

 
Њ "ZЂW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ д
5__inference_material_encoder_11_layer_call_fn_6790783 !"3Ђ0
)Ђ&
 
inputsџџџџџџџџџS

 
Њ "ZЂW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ 
%__inference_signature_wrapper_6790362o !"/Ђ,
Ђ 
%Њ"
 
x
xџџџџџџџџџS"3Њ0
.
output_0"
output_0џџџџџџџџџ Ј
L__inference_unify_vector_14_layer_call_and_return_conditional_losses_6791006X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
1__inference_unify_vector_14_layer_call_fn_6790995K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Љ
J__inference_zero_shift_11_layer_call_and_return_conditional_losses_6790958[/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "%Ђ"

0џџџџџџџџџS
 
/__inference_zero_shift_11_layer_call_fn_6790947N/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "џџџџџџџџџS