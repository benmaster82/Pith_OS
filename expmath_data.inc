; Eight-lane constants of expmath.asm.
align 32
%macro LANES 2
%1: times 8 dd %2
%endmacro
LANES exp_min8,-80.0
LANES exp_max8,80.0
LANES log2e8,1.44269504088896341
LANES ln2_hi8,0.693359375
LANES ln2_lo8,-2.12194440e-4
LANES exp_p0,1.9875691500e-4
LANES exp_p1,1.3981999507e-3
LANES exp_p2,8.3334519073e-3
LANES exp_p3,4.1665795894e-2
LANES exp_p4,1.6666665459e-1
LANES exp_p5,5.0000001201e-1
LANES one8,1.0
LANES exp_bias8,127
lane_index8: dd 0,1,2,3,4,5,6,7
