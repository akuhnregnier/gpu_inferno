#ifndef inferno_hpp
#define inferno_hpp

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>
#include <stdexcept>
#include <tuple>

#include "common.hpp"
#include "loadLibrary.hpp"

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;


const float es[1553] = {
    0.966483e-02, 0.966483e-02, 0.984279e-02, 0.100240e-01, 0.102082e-01,
    0.103957e-01, 0.105865e-01, 0.107803e-01, 0.109777e-01, 0.111784e-01,
    0.113825e-01, 0.115902e-01, 0.118016e-01, 0.120164e-01, 0.122348e-01,
    0.124572e-01, 0.126831e-01, 0.129132e-01, 0.131470e-01, 0.133846e-01,
    0.136264e-01, 0.138724e-01, 0.141225e-01, 0.143771e-01, 0.146356e-01,
    0.148985e-01, 0.151661e-01, 0.154379e-01, 0.157145e-01, 0.159958e-01,
    0.162817e-01, 0.165725e-01, 0.168680e-01, 0.171684e-01, 0.174742e-01,
    0.177847e-01, 0.181008e-01, 0.184216e-01, 0.187481e-01, 0.190801e-01,
    0.194175e-01, 0.197608e-01, 0.201094e-01, 0.204637e-01, 0.208242e-01,
    0.211906e-01, 0.215631e-01, 0.219416e-01, 0.223263e-01, 0.227172e-01,
    0.231146e-01, 0.235188e-01, 0.239296e-01, 0.243465e-01, 0.247708e-01,
    0.252019e-01, 0.256405e-01, 0.260857e-01, 0.265385e-01, 0.269979e-01,
    0.274656e-01, 0.279405e-01, 0.284232e-01, 0.289142e-01, 0.294124e-01,
    0.299192e-01, 0.304341e-01, 0.309571e-01, 0.314886e-01, 0.320285e-01,
    0.325769e-01, 0.331348e-01, 0.337014e-01, 0.342771e-01, 0.348618e-01,
    0.354557e-01, 0.360598e-01, 0.366727e-01, 0.372958e-01, 0.379289e-01,
    0.385717e-01, 0.392248e-01, 0.398889e-01, 0.405633e-01, 0.412474e-01,
    0.419430e-01, 0.426505e-01, 0.433678e-01, 0.440974e-01, 0.448374e-01,
    0.455896e-01, 0.463545e-01, 0.471303e-01, 0.479191e-01, 0.487190e-01,
    0.495322e-01, 0.503591e-01, 0.511977e-01, 0.520490e-01, 0.529145e-01,
    0.537931e-01, 0.546854e-01, 0.555924e-01, 0.565119e-01, 0.574467e-01,
    0.583959e-01, 0.593592e-01, 0.603387e-01, 0.613316e-01, 0.623409e-01,
    0.633655e-01, 0.644053e-01, 0.654624e-01, 0.665358e-01, 0.676233e-01,
    0.687302e-01, 0.698524e-01, 0.709929e-01, 0.721490e-01, 0.733238e-01,
    0.745180e-01, 0.757281e-01, 0.769578e-01, 0.782061e-01, 0.794728e-01,
    0.807583e-01, 0.820647e-01, 0.833905e-01, 0.847358e-01, 0.861028e-01,
    0.874882e-01, 0.888957e-01, 0.903243e-01, 0.917736e-01, 0.932464e-01,
    0.947407e-01, 0.962571e-01, 0.977955e-01, 0.993584e-01, 0.100942e00,
    0.102551e00, 0.104186e00, 0.105842e00, 0.107524e00, 0.109231e00,
    0.110963e00, 0.112722e00, 0.114506e00, 0.116317e00, 0.118153e00,
    0.120019e00, 0.121911e00, 0.123831e00, 0.125778e00, 0.127755e00,
    0.129761e00, 0.131796e00, 0.133863e00, 0.135956e00, 0.138082e00,
    0.140241e00, 0.142428e00, 0.144649e00, 0.146902e00, 0.149190e00,
    0.151506e00, 0.153859e00, 0.156245e00, 0.158669e00, 0.161126e00,
    0.163618e00, 0.166145e00, 0.168711e00, 0.171313e00, 0.173951e00,
    0.176626e00, 0.179342e00, 0.182096e00, 0.184893e00, 0.187724e00,
    0.190600e00, 0.193518e00, 0.196473e00, 0.199474e00, 0.202516e00,
    0.205604e00, 0.208730e00, 0.211905e00, 0.215127e00, 0.218389e00,
    0.221701e00, 0.225063e00, 0.228466e00, 0.231920e00, 0.235421e00,
    0.238976e00, 0.242580e00, 0.246232e00, 0.249933e00, 0.253691e00,
    0.257499e00, 0.261359e00, 0.265278e00, 0.269249e00, 0.273274e00,
    0.277358e00, 0.281498e00, 0.285694e00, 0.289952e00, 0.294268e00,
    0.298641e00, 0.303078e00, 0.307577e00, 0.312135e00, 0.316753e00,
    0.321440e00, 0.326196e00, 0.331009e00, 0.335893e00, 0.340842e00,
    0.345863e00, 0.350951e00, 0.356106e00, 0.361337e00, 0.366636e00,
    0.372006e00, 0.377447e00, 0.382966e00, 0.388567e00, 0.394233e00,
    0.399981e00, 0.405806e00, 0.411714e00, 0.417699e00, 0.423772e00,
    0.429914e00, 0.436145e00, 0.442468e00, 0.448862e00, 0.455359e00,
    0.461930e00, 0.468596e00, 0.475348e00, 0.482186e00, 0.489124e00,
    0.496160e00, 0.503278e00, 0.510497e00, 0.517808e00, 0.525224e00,
    0.532737e00, 0.540355e00, 0.548059e00, 0.555886e00, 0.563797e00,
    0.571825e00, 0.579952e00, 0.588198e00, 0.596545e00, 0.605000e00,
    0.613572e00, 0.622255e00, 0.631059e00, 0.639962e00, 0.649003e00,
    0.658144e00, 0.667414e00, 0.676815e00, 0.686317e00, 0.695956e00,
    0.705728e00, 0.715622e00, 0.725641e00, 0.735799e00, 0.746082e00,
    0.756495e00, 0.767052e00, 0.777741e00, 0.788576e00, 0.799549e00,
    0.810656e00, 0.821914e00, 0.833314e00, 0.844854e00, 0.856555e00,
    0.868415e00, 0.880404e00, 0.892575e00, 0.904877e00, 0.917350e00,
    0.929974e00, 0.942771e00, 0.955724e00, 0.968837e00, 0.982127e00,
    0.995600e00, 0.100921e01, 0.102304e01, 0.103700e01, 0.105116e01,
    0.106549e01, 0.108002e01, 0.109471e01, 0.110962e01, 0.112469e01,
    0.113995e01, 0.115542e01, 0.117107e01, 0.118693e01, 0.120298e01,
    0.121923e01, 0.123569e01, 0.125234e01, 0.126923e01, 0.128631e01,
    0.130362e01, 0.132114e01, 0.133887e01, 0.135683e01, 0.137500e01,
    0.139342e01, 0.141205e01, 0.143091e01, 0.145000e01, 0.146933e01,
    0.148892e01, 0.150874e01, 0.152881e01, 0.154912e01, 0.156970e01,
    0.159049e01, 0.161159e01, 0.163293e01, 0.165452e01, 0.167640e01,
    0.169852e01, 0.172091e01, 0.174359e01, 0.176653e01, 0.178977e01,
    0.181332e01, 0.183709e01, 0.186119e01, 0.188559e01, 0.191028e01,
    0.193524e01, 0.196054e01, 0.198616e01, 0.201208e01, 0.203829e01,
    0.206485e01, 0.209170e01, 0.211885e01, 0.214637e01, 0.217424e01,
    0.220242e01, 0.223092e01, 0.225979e01, 0.228899e01, 0.231855e01,
    0.234845e01, 0.237874e01, 0.240937e01, 0.244040e01, 0.247176e01,
    0.250349e01, 0.253560e01, 0.256814e01, 0.260099e01, 0.263431e01,
    0.266800e01, 0.270207e01, 0.273656e01, 0.277145e01, 0.280671e01,
    0.284248e01, 0.287859e01, 0.291516e01, 0.295219e01, 0.298962e01,
    0.302746e01, 0.306579e01, 0.310454e01, 0.314377e01, 0.318351e01,
    0.322360e01, 0.326427e01, 0.330538e01, 0.334694e01, 0.338894e01,
    0.343155e01, 0.347456e01, 0.351809e01, 0.356216e01, 0.360673e01,
    0.365184e01, 0.369744e01, 0.374352e01, 0.379018e01, 0.383743e01,
    0.388518e01, 0.393344e01, 0.398230e01, 0.403177e01, 0.408175e01,
    0.413229e01, 0.418343e01, 0.423514e01, 0.428746e01, 0.434034e01,
    0.439389e01, 0.444808e01, 0.450276e01, 0.455820e01, 0.461423e01,
    0.467084e01, 0.472816e01, 0.478607e01, 0.484468e01, 0.490393e01,
    0.496389e01, 0.502446e01, 0.508580e01, 0.514776e01, 0.521047e01,
    0.527385e01, 0.533798e01, 0.540279e01, 0.546838e01, 0.553466e01,
    0.560173e01, 0.566949e01, 0.573807e01, 0.580750e01, 0.587749e01,
    0.594846e01, 0.602017e01, 0.609260e01, 0.616591e01, 0.623995e01,
    0.631490e01, 0.639061e01, 0.646723e01, 0.654477e01, 0.662293e01,
    0.670220e01, 0.678227e01, 0.686313e01, 0.694495e01, 0.702777e01,
    0.711142e01, 0.719592e01, 0.728140e01, 0.736790e01, 0.745527e01,
    0.754352e01, 0.763298e01, 0.772316e01, 0.781442e01, 0.790676e01,
    0.800001e01, 0.809435e01, 0.818967e01, 0.828606e01, 0.838343e01,
    0.848194e01, 0.858144e01, 0.868207e01, 0.878392e01, 0.888673e01,
    0.899060e01, 0.909567e01, 0.920172e01, 0.930909e01, 0.941765e01,
    0.952730e01, 0.963821e01, 0.975022e01, 0.986352e01, 0.997793e01,
    0.100937e02, 0.102105e02, 0.103287e02, 0.104481e02, 0.105688e02,
    0.106909e02, 0.108143e02, 0.109387e02, 0.110647e02, 0.111921e02,
    0.113207e02, 0.114508e02, 0.115821e02, 0.117149e02, 0.118490e02,
    0.119847e02, 0.121216e02, 0.122601e02, 0.124002e02, 0.125416e02,
    0.126846e02, 0.128290e02, 0.129747e02, 0.131224e02, 0.132712e02,
    0.134220e02, 0.135742e02, 0.137278e02, 0.138831e02, 0.140403e02,
    0.141989e02, 0.143589e02, 0.145211e02, 0.146845e02, 0.148501e02,
    0.150172e02, 0.151858e02, 0.153564e02, 0.155288e02, 0.157029e02,
    0.158786e02, 0.160562e02, 0.162358e02, 0.164174e02, 0.166004e02,
    0.167858e02, 0.169728e02, 0.171620e02, 0.173528e02, 0.175455e02,
    0.177406e02, 0.179372e02, 0.181363e02, 0.183372e02, 0.185400e02,
    0.187453e02, 0.189523e02, 0.191613e02, 0.193728e02, 0.195866e02,
    0.198024e02, 0.200200e02, 0.202401e02, 0.204626e02, 0.206871e02,
    0.209140e02, 0.211430e02, 0.213744e02, 0.216085e02, 0.218446e02,
    0.220828e02, 0.223241e02, 0.225671e02, 0.228132e02, 0.230615e02,
    0.233120e02, 0.235651e02, 0.238211e02, 0.240794e02, 0.243404e02,
    0.246042e02, 0.248704e02, 0.251390e02, 0.254109e02, 0.256847e02,
    0.259620e02, 0.262418e02, 0.265240e02, 0.268092e02, 0.270975e02,
    0.273883e02, 0.276822e02, 0.279792e02, 0.282789e02, 0.285812e02,
    0.288867e02, 0.291954e02, 0.295075e02, 0.298222e02, 0.301398e02,
    0.304606e02, 0.307848e02, 0.311119e02, 0.314424e02, 0.317763e02,
    0.321133e02, 0.324536e02, 0.327971e02, 0.331440e02, 0.334940e02,
    0.338475e02, 0.342050e02, 0.345654e02, 0.349295e02, 0.352975e02,
    0.356687e02, 0.360430e02, 0.364221e02, 0.368042e02, 0.371896e02,
    0.375790e02, 0.379725e02, 0.383692e02, 0.387702e02, 0.391744e02,
    0.395839e02, 0.399958e02, 0.404118e02, 0.408325e02, 0.412574e02,
    0.416858e02, 0.421188e02, 0.425551e02, 0.429962e02, 0.434407e02,
    0.438910e02, 0.443439e02, 0.448024e02, 0.452648e02, 0.457308e02,
    0.462018e02, 0.466775e02, 0.471582e02, 0.476428e02, 0.481313e02,
    0.486249e02, 0.491235e02, 0.496272e02, 0.501349e02, 0.506479e02,
    0.511652e02, 0.516876e02, 0.522142e02, 0.527474e02, 0.532836e02,
    0.538266e02, 0.543737e02, 0.549254e02, 0.554839e02, 0.560456e02,
    0.566142e02, 0.571872e02, 0.577662e02, 0.583498e02, 0.589392e02,
    0.595347e02, 0.601346e02, 0.607410e02, 0.613519e02, 0.619689e02,
    0.625922e02, 0.632204e02, 0.638550e02, 0.644959e02, 0.651418e02,
    0.657942e02, 0.664516e02, 0.671158e02, 0.677864e02, 0.684624e02,
    0.691451e02, 0.698345e02, 0.705293e02, 0.712312e02, 0.719398e02,
    0.726542e02, 0.733754e02, 0.741022e02, 0.748363e02, 0.755777e02,
    0.763247e02, 0.770791e02, 0.778394e02, 0.786088e02, 0.793824e02,
    0.801653e02, 0.809542e02, 0.817509e02, 0.825536e02, 0.833643e02,
    0.841828e02, 0.850076e02, 0.858405e02, 0.866797e02, 0.875289e02,
    0.883827e02, 0.892467e02, 0.901172e02, 0.909962e02, 0.918818e02,
    0.927760e02, 0.936790e02, 0.945887e02, 0.955071e02, 0.964346e02,
    0.973689e02, 0.983123e02, 0.992648e02, 0.100224e03, 0.101193e03,
    0.102169e03, 0.103155e03, 0.104150e03, 0.105152e03, 0.106164e03,
    0.107186e03, 0.108217e03, 0.109256e03, 0.110303e03, 0.111362e03,
    0.112429e03, 0.113503e03, 0.114588e03, 0.115684e03, 0.116789e03,
    0.117903e03, 0.119028e03, 0.120160e03, 0.121306e03, 0.122460e03,
    0.123623e03, 0.124796e03, 0.125981e03, 0.127174e03, 0.128381e03,
    0.129594e03, 0.130822e03, 0.132058e03, 0.133306e03, 0.134563e03,
    0.135828e03, 0.137109e03, 0.138402e03, 0.139700e03, 0.141017e03,
    0.142338e03, 0.143676e03, 0.145025e03, 0.146382e03, 0.147753e03,
    0.149133e03, 0.150529e03, 0.151935e03, 0.153351e03, 0.154783e03,
    0.156222e03, 0.157678e03, 0.159148e03, 0.160624e03, 0.162117e03,
    0.163621e03, 0.165142e03, 0.166674e03, 0.168212e03, 0.169772e03,
    0.171340e03, 0.172921e03, 0.174522e03, 0.176129e03, 0.177755e03,
    0.179388e03, 0.181040e03, 0.182707e03, 0.184382e03, 0.186076e03,
    0.187782e03, 0.189503e03, 0.191240e03, 0.192989e03, 0.194758e03,
    0.196535e03, 0.198332e03, 0.200141e03, 0.201963e03, 0.203805e03,
    0.205656e03, 0.207532e03, 0.209416e03, 0.211317e03, 0.213236e03,
    0.215167e03, 0.217121e03, 0.219087e03, 0.221067e03, 0.223064e03,
    0.225080e03, 0.227113e03, 0.229160e03, 0.231221e03, 0.233305e03,
    0.235403e03, 0.237520e03, 0.239655e03, 0.241805e03, 0.243979e03,
    0.246163e03, 0.248365e03, 0.250593e03, 0.252830e03, 0.255093e03,
    0.257364e03, 0.259667e03, 0.261979e03, 0.264312e03, 0.266666e03,
    0.269034e03, 0.271430e03, 0.273841e03, 0.276268e03, 0.278722e03,
    0.281185e03, 0.283677e03, 0.286190e03, 0.288714e03, 0.291266e03,
    0.293834e03, 0.296431e03, 0.299045e03, 0.301676e03, 0.304329e03,
    0.307006e03, 0.309706e03, 0.312423e03, 0.315165e03, 0.317930e03,
    0.320705e03, 0.323519e03, 0.326350e03, 0.329199e03, 0.332073e03,
    0.334973e03, 0.337897e03, 0.340839e03, 0.343800e03, 0.346794e03,
    0.349806e03, 0.352845e03, 0.355918e03, 0.358994e03, 0.362112e03,
    0.365242e03, 0.368407e03, 0.371599e03, 0.374802e03, 0.378042e03,
    0.381293e03, 0.384588e03, 0.387904e03, 0.391239e03, 0.394604e03,
    0.397988e03, 0.401411e03, 0.404862e03, 0.408326e03, 0.411829e03,
    0.415352e03, 0.418906e03, 0.422490e03, 0.426095e03, 0.429740e03,
    0.433398e03, 0.437097e03, 0.440827e03, 0.444570e03, 0.448354e03,
    0.452160e03, 0.455999e03, 0.459870e03, 0.463765e03, 0.467702e03,
    0.471652e03, 0.475646e03, 0.479674e03, 0.483715e03, 0.487811e03,
    0.491911e03, 0.496065e03, 0.500244e03, 0.504448e03, 0.508698e03,
    0.512961e03, 0.517282e03, 0.521617e03, 0.525989e03, 0.530397e03,
    0.534831e03, 0.539313e03, 0.543821e03, 0.548355e03, 0.552938e03,
    0.557549e03, 0.562197e03, 0.566884e03, 0.571598e03, 0.576351e03,
    0.581131e03, 0.585963e03, 0.590835e03, 0.595722e03, 0.600663e03,
    0.605631e03, 0.610641e03, 0.615151e03, 0.619625e03, 0.624140e03,
    0.628671e03, 0.633243e03, 0.637845e03, 0.642465e03, 0.647126e03,
    0.651806e03, 0.656527e03, 0.661279e03, 0.666049e03, 0.670861e03,
    0.675692e03, 0.680566e03, 0.685471e03, 0.690396e03, 0.695363e03,
    0.700350e03, 0.705381e03, 0.710444e03, 0.715527e03, 0.720654e03,
    0.725801e03, 0.730994e03, 0.736219e03, 0.741465e03, 0.746756e03,
    0.752068e03, 0.757426e03, 0.762819e03, 0.768231e03, 0.773692e03,
    0.779172e03, 0.784701e03, 0.790265e03, 0.795849e03, 0.801483e03,
    0.807137e03, 0.812842e03, 0.818582e03, 0.824343e03, 0.830153e03,
    0.835987e03, 0.841871e03, 0.847791e03, 0.853733e03, 0.859727e03,
    0.865743e03, 0.871812e03, 0.877918e03, 0.884046e03, 0.890228e03,
    0.896433e03, 0.902690e03, 0.908987e03, 0.915307e03, 0.921681e03,
    0.928078e03, 0.934531e03, 0.941023e03, 0.947539e03, 0.954112e03,
    0.960708e03, 0.967361e03, 0.974053e03, 0.980771e03, 0.987545e03,
    0.994345e03, 0.100120e04, 0.100810e04, 0.101502e04, 0.102201e04,
    0.102902e04, 0.103608e04, 0.104320e04, 0.105033e04, 0.105753e04,
    0.106475e04, 0.107204e04, 0.107936e04, 0.108672e04, 0.109414e04,
    0.110158e04, 0.110908e04, 0.111663e04, 0.112421e04, 0.113185e04,
    0.113952e04, 0.114725e04, 0.115503e04, 0.116284e04, 0.117071e04,
    0.117861e04, 0.118658e04, 0.119459e04, 0.120264e04, 0.121074e04,
    0.121888e04, 0.122709e04, 0.123534e04, 0.124362e04, 0.125198e04,
    0.126036e04, 0.126881e04, 0.127731e04, 0.128584e04, 0.129444e04,
    0.130307e04, 0.131177e04, 0.132053e04, 0.132931e04, 0.133817e04,
    0.134705e04, 0.135602e04, 0.136503e04, 0.137407e04, 0.138319e04,
    0.139234e04, 0.140156e04, 0.141084e04, 0.142015e04, 0.142954e04,
    0.143896e04, 0.144845e04, 0.145800e04, 0.146759e04, 0.147725e04,
    0.148694e04, 0.149672e04, 0.150655e04, 0.151641e04, 0.152635e04,
    0.153633e04, 0.154639e04, 0.155650e04, 0.156665e04, 0.157688e04,
    0.158715e04, 0.159750e04, 0.160791e04, 0.161836e04, 0.162888e04,
    0.163945e04, 0.165010e04, 0.166081e04, 0.167155e04, 0.168238e04,
    0.169325e04, 0.170420e04, 0.171522e04, 0.172627e04, 0.173741e04,
    0.174859e04, 0.175986e04, 0.177119e04, 0.178256e04, 0.179402e04,
    0.180552e04, 0.181711e04, 0.182877e04, 0.184046e04, 0.185224e04,
    0.186407e04, 0.187599e04, 0.188797e04, 0.190000e04, 0.191212e04,
    0.192428e04, 0.193653e04, 0.194886e04, 0.196122e04, 0.197368e04,
    0.198618e04, 0.199878e04, 0.201145e04, 0.202416e04, 0.203698e04,
    0.204983e04, 0.206278e04, 0.207580e04, 0.208887e04, 0.210204e04,
    0.211525e04, 0.212856e04, 0.214195e04, 0.215538e04, 0.216892e04,
    0.218249e04, 0.219618e04, 0.220994e04, 0.222375e04, 0.223766e04,
    0.225161e04, 0.226567e04, 0.227981e04, 0.229399e04, 0.230829e04,
    0.232263e04, 0.233708e04, 0.235161e04, 0.236618e04, 0.238087e04,
    0.239560e04, 0.241044e04, 0.242538e04, 0.244035e04, 0.245544e04,
    0.247057e04, 0.248583e04, 0.250116e04, 0.251654e04, 0.253204e04,
    0.254759e04, 0.256325e04, 0.257901e04, 0.259480e04, 0.261073e04,
    0.262670e04, 0.264279e04, 0.265896e04, 0.267519e04, 0.269154e04,
    0.270794e04, 0.272447e04, 0.274108e04, 0.275774e04, 0.277453e04,
    0.279137e04, 0.280834e04, 0.282540e04, 0.284251e04, 0.285975e04,
    0.287704e04, 0.289446e04, 0.291198e04, 0.292954e04, 0.294725e04,
    0.296499e04, 0.298288e04, 0.300087e04, 0.301890e04, 0.303707e04,
    0.305529e04, 0.307365e04, 0.309211e04, 0.311062e04, 0.312927e04,
    0.314798e04, 0.316682e04, 0.318577e04, 0.320477e04, 0.322391e04,
    0.324310e04, 0.326245e04, 0.328189e04, 0.330138e04, 0.332103e04,
    0.334073e04, 0.336058e04, 0.338053e04, 0.340054e04, 0.342069e04,
    0.344090e04, 0.346127e04, 0.348174e04, 0.350227e04, 0.352295e04,
    0.354369e04, 0.356458e04, 0.358559e04, 0.360664e04, 0.362787e04,
    0.364914e04, 0.367058e04, 0.369212e04, 0.371373e04, 0.373548e04,
    0.375731e04, 0.377929e04, 0.380139e04, 0.382355e04, 0.384588e04,
    0.386826e04, 0.389081e04, 0.391348e04, 0.393620e04, 0.395910e04,
    0.398205e04, 0.400518e04, 0.402843e04, 0.405173e04, 0.407520e04,
    0.409875e04, 0.412246e04, 0.414630e04, 0.417019e04, 0.419427e04,
    0.421840e04, 0.424272e04, 0.426715e04, 0.429165e04, 0.431634e04,
    0.434108e04, 0.436602e04, 0.439107e04, 0.441618e04, 0.444149e04,
    0.446685e04, 0.449241e04, 0.451810e04, 0.454385e04, 0.456977e04,
    0.459578e04, 0.462197e04, 0.464830e04, 0.467468e04, 0.470127e04,
    0.472792e04, 0.475477e04, 0.478175e04, 0.480880e04, 0.483605e04,
    0.486336e04, 0.489087e04, 0.491853e04, 0.494623e04, 0.497415e04,
    0.500215e04, 0.503034e04, 0.505867e04, 0.508707e04, 0.511568e04,
    0.514436e04, 0.517325e04, 0.520227e04, 0.523137e04, 0.526068e04,
    0.529005e04, 0.531965e04, 0.534939e04, 0.537921e04, 0.540923e04,
    0.543932e04, 0.546965e04, 0.550011e04, 0.553064e04, 0.556139e04,
    0.559223e04, 0.562329e04, 0.565449e04, 0.568577e04, 0.571727e04,
    0.574884e04, 0.578064e04, 0.581261e04, 0.584464e04, 0.587692e04,
    0.590924e04, 0.594182e04, 0.597455e04, 0.600736e04, 0.604039e04,
    0.607350e04, 0.610685e04, 0.614036e04, 0.617394e04, 0.620777e04,
    0.624169e04, 0.627584e04, 0.631014e04, 0.634454e04, 0.637918e04,
    0.641390e04, 0.644887e04, 0.648400e04, 0.651919e04, 0.655467e04,
    0.659021e04, 0.662599e04, 0.666197e04, 0.669800e04, 0.673429e04,
    0.677069e04, 0.680735e04, 0.684415e04, 0.688104e04, 0.691819e04,
    0.695543e04, 0.699292e04, 0.703061e04, 0.706837e04, 0.710639e04,
    0.714451e04, 0.718289e04, 0.722143e04, 0.726009e04, 0.729903e04,
    0.733802e04, 0.737729e04, 0.741676e04, 0.745631e04, 0.749612e04,
    0.753602e04, 0.757622e04, 0.761659e04, 0.765705e04, 0.769780e04,
    0.773863e04, 0.777975e04, 0.782106e04, 0.786246e04, 0.790412e04,
    0.794593e04, 0.798802e04, 0.803028e04, 0.807259e04, 0.811525e04,
    0.815798e04, 0.820102e04, 0.824427e04, 0.828757e04, 0.833120e04,
    0.837493e04, 0.841895e04, 0.846313e04, 0.850744e04, 0.855208e04,
    0.859678e04, 0.864179e04, 0.868705e04, 0.873237e04, 0.877800e04,
    0.882374e04, 0.886979e04, 0.891603e04, 0.896237e04, 0.900904e04,
    0.905579e04, 0.910288e04, 0.915018e04, 0.919758e04, 0.924529e04,
    0.929310e04, 0.934122e04, 0.938959e04, 0.943804e04, 0.948687e04,
    0.953575e04, 0.958494e04, 0.963442e04, 0.968395e04, 0.973384e04,
    0.978383e04, 0.983412e04, 0.988468e04, 0.993534e04, 0.998630e04,
    0.100374e05, 0.100888e05, 0.101406e05, 0.101923e05, 0.102444e05,
    0.102966e05, 0.103492e05, 0.104020e05, 0.104550e05, 0.105082e05,
    0.105616e05, 0.106153e05, 0.106693e05, 0.107234e05, 0.107779e05,
    0.108325e05, 0.108874e05, 0.109425e05, 0.109978e05, 0.110535e05,
    0.111092e05, 0.111653e05, 0.112217e05, 0.112782e05, 0.113350e05,
    0.113920e05, 0.114493e05, 0.115070e05, 0.115646e05, 0.116228e05,
    0.116809e05, 0.117396e05, 0.117984e05, 0.118574e05, 0.119167e05,
    0.119762e05, 0.120360e05, 0.120962e05, 0.121564e05, 0.122170e05,
    0.122778e05, 0.123389e05, 0.124004e05, 0.124619e05, 0.125238e05,
    0.125859e05, 0.126484e05, 0.127111e05, 0.127739e05, 0.128372e05,
    0.129006e05, 0.129644e05, 0.130285e05, 0.130927e05, 0.131573e05,
    0.132220e05, 0.132872e05, 0.133526e05, 0.134182e05, 0.134842e05,
    0.135503e05, 0.136168e05, 0.136836e05, 0.137505e05, 0.138180e05,
    0.138854e05, 0.139534e05, 0.140216e05, 0.140900e05, 0.141588e05,
    0.142277e05, 0.142971e05, 0.143668e05, 0.144366e05, 0.145069e05,
    0.145773e05, 0.146481e05, 0.147192e05, 0.147905e05, 0.148622e05,
    0.149341e05, 0.150064e05, 0.150790e05, 0.151517e05, 0.152250e05,
    0.152983e05, 0.153721e05, 0.154462e05, 0.155205e05, 0.155952e05,
    0.156701e05, 0.157454e05, 0.158211e05, 0.158969e05, 0.159732e05,
    0.160496e05, 0.161265e05, 0.162037e05, 0.162811e05, 0.163589e05,
    0.164369e05, 0.165154e05, 0.165942e05, 0.166732e05, 0.167526e05,
    0.168322e05, 0.169123e05, 0.169927e05, 0.170733e05, 0.171543e05,
    0.172356e05, 0.173173e05, 0.173993e05, 0.174815e05, 0.175643e05,
    0.176471e05, 0.177305e05, 0.178143e05, 0.178981e05, 0.179826e05,
    0.180671e05, 0.181522e05, 0.182377e05, 0.183232e05, 0.184093e05,
    0.184955e05, 0.185823e05, 0.186695e05, 0.187568e05, 0.188447e05,
    0.189326e05, 0.190212e05, 0.191101e05, 0.191991e05, 0.192887e05,
    0.193785e05, 0.194688e05, 0.195595e05, 0.196503e05, 0.197417e05,
    0.198332e05, 0.199253e05, 0.200178e05, 0.201105e05, 0.202036e05,
    0.202971e05, 0.203910e05, 0.204853e05, 0.205798e05, 0.206749e05,
    0.207701e05, 0.208659e05, 0.209621e05, 0.210584e05, 0.211554e05,
    0.212524e05, 0.213501e05, 0.214482e05, 0.215465e05, 0.216452e05,
    0.217442e05, 0.218439e05, 0.219439e05, 0.220440e05, 0.221449e05,
    0.222457e05, 0.223473e05, 0.224494e05, 0.225514e05, 0.226542e05,
    0.227571e05, 0.228606e05, 0.229646e05, 0.230687e05, 0.231734e05,
    0.232783e05, 0.233839e05, 0.234898e05, 0.235960e05, 0.237027e05,
    0.238097e05, 0.239173e05, 0.240254e05, 0.241335e05, 0.242424e05,
    0.243514e05, 0.244611e05, 0.245712e05, 0.246814e05, 0.247923e05,
    0.249034e05, 0.250152e05, 0.250152e05
};


float qsat_wat(float t, float p) {
    // Calculates saturation vapour pressure

    // Parameters
    // ----------
    // t : float
    //     Temperature
    // p : float
    //     surface pressure

    // Returns
    // -------
    // float
    //     Saturation vapour pressure
    const float repsilon = 0.62198;
    const float one_minus_epsilon = 1.0 - repsilon;
    const float zerodegc = 273.15;
    const float delta_t = 0.1;
    const float t_low = 183.15;
    const float t_high = 338.15;

    const float one = 1.0;
    const float pconv = 1.0e-8;
    const float term1 = 4.5;
    const float term2 = 6.0e-4;

    float fsubw, tt, qs, float_atable;
    int itable, atable;

    // Compute the factor that converts from sat vapour pressure in a
    // pure water system to sat vapour pressure in air, fsubw.
    // This formula is taken from equation A4.7 of Adrian Gill's book:
    // atmosphere-ocean dynamics. Note that his formula works in terms
    // of pressure in mb and temperature in celsius, so conversion of
    // units leads to the slightly different equation used here.
    fsubw = one + pconv * p * (term1 + term2 * (t - zerodegc) * (t - zerodegc));

    // Use the lookup table to find saturated vapour pressure. Store it in qs.
    tt = std::max(t_low, t);
    tt = std::min(t_high, tt);
    float_atable = (tt - t_low + delta_t) / delta_t;
    itable = static_cast<int>(floor(float_atable));
    atable = static_cast<int>(floor(float_atable - itable));
    qs = (one - atable) * es[itable] + atable * es[itable + 1];

    // Multiply by fsubw to convert to saturated vapour pressure in air
    // (equation A4.6 OF Adrian Gill's book).
    qs *= fsubw;

    // Now form the accurate expression for qs, which is a rearranged
    // version of equation A4.3 of Gill's book.
    // Note that at very low pressures we apply a fix, to prevent a
    // singularity (qsat tends to 1. kg/kg).
    return (repsilon * qs) / (std::max(p, qs) - one_minus_epsilon * qs);
}


inline int get_index_3d(int x, int y, int z, const int* shape_3d) {
    return (x * shape_3d[1] * shape_3d[2]) + (y * shape_3d[2]) + z;
}


inline int get_index_2d(int x, int y, const int* shape_2d) {
    return (x * shape_2d[1]) + y;
}


void calculateDiagnostics(
    int Nt,
    int landPts,
    int nPFT,
    int nTotalPFT,
    const float* t1p5m_tile,
    const float* q1p5m_tile,
    const float* pstar,
    const float* ls_rain,
    const float* con_rain,
    float* inferno_rain_out,  // (Nt, land_pts)
    float* qsat_out,  // (Nt, npft, land_pts)
    float* inferno_rhum_out  // (Nt, npft, land_pts)
) {
    // Tolerance number to filter non-physical rain values
    const float rain_tolerance = 1.0e-18;  // kg/m2/s

    const int shape_2d[2] = { Nt, landPts };
    const int total_pft_shape_3d[3] = { Nt, nTotalPFT, landPts };
    const int nat_pft_shape_3d[3] = { Nt, nPFT, landPts };

    for (int ti = 0; ti < Nt; ti++) {
        for (int li = 0; li < landPts; li++) {

            int _2d_flat_i = get_index_2d(ti, li, shape_2d);

            float ls_rain_val = ls_rain[_2d_flat_i];
            float con_rain_val = con_rain[_2d_flat_i];
            float pstar_val = pstar[_2d_flat_i];

            if (ls_rain_val < rain_tolerance) {
                ls_rain_val = 0.0;
            }
            if (con_rain_val < rain_tolerance) {
                con_rain_val = 0.0;
            }

            float inferno_rain_l = ls_rain_val + con_rain_val;
            inferno_rain_out[_2d_flat_i] = inferno_rain_l;

            for (int pfti = 0; pfti < nPFT; pfti++) {
                int nat_pft_3d_flat_i = get_index_3d(ti, pfti, li, nat_pft_shape_3d);
                int total_pft_3d_flat_i = get_index_3d(ti, pfti, li, total_pft_shape_3d);
                float temperature = t1p5m_tile[total_pft_3d_flat_i];
                float q1p5m_tile_val = q1p5m_tile[total_pft_3d_flat_i];

                float qsat_l = qsat_wat(temperature, pstar_val);
                float inferno_rhum_l = (q1p5m_tile_val / qsat_l) * 100.0;

                qsat_out[nat_pft_3d_flat_i] = qsat_l;
                inferno_rhum_out[nat_pft_3d_flat_i] = inferno_rhum_l;
            }
        }
    }
}


class GPUInfernoBase : public GPUBase {

public:
    static const int nData = 17;
    static const int nParam = 22;

    MTL::Buffer* outputBuffer;
    std::array<MTL::Buffer*, nData> dataBuffers;
    std::array<MTL::Buffer*, nParam> paramBuffers;
    MTL::Buffer* checksFailedBuffer;

    // Parameters.
    int ignitionMethod;
    int flammabilityMethod;
    int drynessMethod;
    int fuelBuildUpMethod;
    int includeTemperature;
    int Nt;

    MTL::ArgumentEncoder* argumentEncoder;
    MTL::Buffer* argumentBuffer;

    // Misc.
    bool didSetData = false;
    bool didSetParams = false;

    MTL::Buffer* createBufferFromPyArray(pyArray array) {
        py::buffer_info buf = array.request();
        return device->newBuffer(static_cast<float *>(buf.ptr), buf.shape[0] * dataSize, MTL::ResourceOptions());
    }

    GPUInfernoBase(
        MTL::Library* (*loadLibraryFunc)(MTL::Device*),
        const std::string& kernelName
    ) : GPUBase(
        loadLibraryFunc,
        kernelName
    ) {
        // Create parameter buffers for later reuse.
        for (unsigned long i = 0; i < nParam; i++) {
            paramBuffers[i] = device->newBuffer(nPFTGroups * dataSize, MTL::ResourceOptions());
            releaseLater(paramBuffers[i]);
        }
    }

    virtual void setOutputBuffer() = 0;

    virtual void checkNt() = 0;

    void set_data(
        int ignitionMethod,
        int flammabilityMethod,
        int drynessMethod,
        int fuelBuildUpMethod,
        int includeTemperature,
        int Nt,  // i.e. <data>.shape[0].
        // -------------- Start data arrays --------------
        // Input arrays.
        pyArray t1p5m_tile,
        pyArray q1p5m_tile,
        pyArray pstar,
        pyArray sthu_soilt_single,
        pyArray frac,
        pyArray c_soil_dpm_gb,
        pyArray c_soil_rpm_gb,
        pyArray canht,
        pyArray ls_rain,
        pyArray con_rain,
        pyArray pop_den,
        pyArray flash_rate,
        pyArray fuel_build_up,
        pyArray fapar_diag_pft,
        pyArray grouped_dry_bal,
        pyArray litter_pool,
        pyArray dry_days,
        pyBoolArray checks_failed
        // -------------- End data arrays --------------
    ) {
        // Set parameters.
        this->ignitionMethod = ignitionMethod;
        this->flammabilityMethod = flammabilityMethod;
        this->drynessMethod = drynessMethod;
        this->fuelBuildUpMethod = fuelBuildUpMethod;
        this->includeTemperature = includeTemperature;
        this->Nt = Nt;

        checkNt();
        if (ignitionMethod != 1)
            throw std::runtime_error("Ignition method has to be 1!");
        if (flammabilityMethod != 2)
            throw std::runtime_error("Flammability method has to be 2!");

        // Create buffers.
        std::array<pyArray, nData> dataArrays = {
            t1p5m_tile,  // 0
            q1p5m_tile,  // 1
            pstar,  // 2
            sthu_soilt_single,  // 3
            frac,  // 4
            c_soil_dpm_gb,  // 5
            c_soil_rpm_gb,  // 6
            canht,  // 7
            ls_rain,  // 8
            con_rain,  // 9
            pop_den,  // 10
            flash_rate,  // 11
            fuel_build_up,  // 12
            fapar_diag_pft,  // 13
            grouped_dry_bal,  // 14
            litter_pool,  // 15
            dry_days,  // 16
        };

        for (unsigned long i = 0; i < nData; i++) {
            dataBuffers[i] = createBufferFromPyArray(dataArrays[i]);
            argumentEncoder->setBuffer(dataBuffers[i], 0, i);
            releaseLater(dataBuffers[i]);
        }

        py::buffer_info checksFailedInfo = checks_failed.request();
        if (checksFailedInfo.size != Nt * nPFT * landPts)
            throw std::runtime_error("Wrong checks_failed size!");

        checksFailedBuffer = device->newBuffer(
            (bool*)checksFailedInfo.ptr,
            Nt * nPFT * landPts * boolSize,
            MTL::ResourceOptions()
        );
        releaseLater(checksFailedBuffer);

        setOutputBuffer();

        didSetData = true;
    }

    pyBoolArray get_checks_failed_mask() {
        if (!(didSetData))
            throw std::runtime_error("Did not set data!");
        return pyBoolArray(Nt * nPFT * landPts, (bool*)checksFailedBuffer->contents());
    }

    std::tuple<pyArray, pyArray, pyArray> get_diagnostics() {
        if (!(didSetData))
            throw std::runtime_error("Did not set data!");

        pyArray inferno_rain_out_arr = pyArray({Nt, landPts});
        pyArray qsat_out_arr = pyArray({Nt, nPFT, landPts});
        pyArray inferno_rhum_out_arr = pyArray({Nt, nPFT, landPts});

        calculateDiagnostics(
            Nt,
            landPts,
            nPFT,
            nTotalPFT,
            (float*)dataBuffers[0]->contents(),
            (float*)dataBuffers[1]->contents(),
            (float*)dataBuffers[2]->contents(),
            (float*)dataBuffers[8]->contents(),
            (float*)dataBuffers[9]->contents(),
            (float*)inferno_rain_out_arr.request().ptr,
            (float*)qsat_out_arr.request().ptr,
            (float*)inferno_rhum_out_arr.request().ptr
        );

        return std::make_tuple(inferno_rain_out_arr, qsat_out_arr, inferno_rhum_out_arr);
    }

    void set_params(
        pyArray fapar_factor,
        pyArray fapar_centre,
        pyArray fapar_shape,
        pyArray fuel_build_up_factor,
        pyArray fuel_build_up_centre,
        pyArray fuel_build_up_shape,
        pyArray temperature_factor,
        pyArray temperature_centre,
        pyArray temperature_shape,
        pyArray dry_day_factor,
        pyArray dry_day_centre,
        pyArray dry_day_shape,
        pyArray dry_bal_factor,
        pyArray dry_bal_centre,
        pyArray dry_bal_shape,
        pyArray litter_pool_factor,
        pyArray litter_pool_centre,
        pyArray litter_pool_shape,
        pyArray fapar_weight,
        pyArray dryness_weight,
        pyArray temperature_weight,
        pyArray fuel_weight
    ) {
        // Create buffers.
        std::array<pyArray, nParam> paramArrays = {
            fapar_factor,
            fapar_centre,
            fapar_shape,
            fuel_build_up_factor,
            fuel_build_up_centre,
            fuel_build_up_shape,
            temperature_factor,
            temperature_centre,
            temperature_shape,
            dry_day_factor,
            dry_day_centre,
            dry_day_shape,
            dry_bal_factor,
            dry_bal_centre,
            dry_bal_shape,
            litter_pool_factor,
            litter_pool_centre,
            litter_pool_shape,
            fapar_weight,
            dryness_weight,
            temperature_weight,
            fuel_weight,
        };

        for (unsigned long i = 0; i < nParam; i++) {
            py::buffer_info paramBuf = paramArrays[i].request();

            assert(paramBuf.shape[0] == nPFTGroups);

            memcpy(paramBuffers[i]->contents(), paramBuf.ptr, nPFTGroups * dataSize);
            paramBuffers[i]->didModifyRange(NS::Range::Make(0, paramBuffers[i]->length()));
        }

        didSetParams = true;
    }
};


class GPUCompute final : public GPUInfernoBase {

public:
    GPUCompute() : GPUInfernoBase(
        loadLibrary,
        "multi_timestep_inferno_ig1_flam2"
    ) {
        argumentEncoder = fn->newArgumentEncoder(7);
        argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceOptions());
        argumentEncoder->setArgumentBuffer(argumentBuffer, 0);

        releaseLater(argumentBuffer);
        releaseLater(argumentEncoder);
    }

    void setOutputBuffer() override {
        if (Nt == 0)
            throw std::runtime_error("Nt is not set (=0).");
        outputBuffer = device->newBuffer(Nt * nPFT * landPts * dataSize, MTL::ResourceOptions());
        releaseLater(outputBuffer);
    }

    void checkNt() override { }

    void run(pyArray out) {
        if (!(didSetData))
            throw std::runtime_error("Did not set data.");
        if (!(didSetParams))
            throw std::runtime_error("Did not set params.");

        py::buffer_info outInfo = out.request();
        if (outInfo.ndim != 2)
            throw std::runtime_error("Incompatible out dimension!");
        if ((outInfo.shape[0] != Nt) || (outInfo.shape[1] != landPts))
            throw std::runtime_error("Expected out dimensions (Nt, landPts)!");

        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        py::gil_scoped_release release;

        // Output.
        computeCommandEncoder->setBuffer(outputBuffer, 0, 0);

        // Parameters.
        computeCommandEncoder->setBytes(&ignitionMethod, paramSize, 1);
        computeCommandEncoder->setBytes(&flammabilityMethod, paramSize, 2);
        computeCommandEncoder->setBytes(&drynessMethod, paramSize, 3);
        computeCommandEncoder->setBytes(&fuelBuildUpMethod, paramSize, 4);
        computeCommandEncoder->setBytes(&includeTemperature, paramSize, 5);
        computeCommandEncoder->setBytes(&Nt, paramSize, 6);

        // Data arrays.
        computeCommandEncoder->setBuffer(argumentBuffer, 0, 7);
        for (unsigned long i = 0; i < nData; i++) {
            computeCommandEncoder->useResource(dataBuffers[i], MTL::ResourceUsageRead);
        }

        // Parameter arrays.
        for (unsigned long i = 0; i < nParam; i++) {
            computeCommandEncoder->setBuffer(paramBuffers[i], 0, 8 + i);
        }

        computeCommandEncoder->setBuffer(checksFailedBuffer, 0, 30);

        submit(Nt * landPts * nPFT, runParams);

        // Acquire GIL before using Python array below.
        py::gil_scoped_acquire acquire;

        // Perform sum over PFT axis.
        //
        // NOTE - The below would be more correct, but does not currently correspond to the
        // NUMBA Python implementation.
        // return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
        //     frac[:, : data.shape[1]], axis=1
        // )
        // return np.sum(data * frac[:, : data.shape[1]], axis=1)

        // Note that multiplication by frac already occurs within the kernel!
        float* ba_frac = (float*)outputBuffer->contents();
        auto outR = out.mutable_unchecked<2>();

        // Sum over PFT axis.
        for (int ti = 0; ti < Nt; ti++) {
            for (int land_i = 0; land_i < landPts; land_i++) {
                float pft_sum = 0.0f;
                for (int pft_i = 0; pft_i < nPFT; pft_i++) {
                    pft_sum += ba_frac[
                        (ti * nPFT * landPts)
                        + (pft_i * landPts)
                        + land_i
                    ];
                }
                outR(ti, land_i) = pft_sum;
            }
        }
    }
};

#endif
