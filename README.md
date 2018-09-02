# Flow Estimation
The purpose of this project is to implement flow estimation with Semi-Global Method. The pixel-wise cost function with Census transformation is involved as same as the stereo implementation. During cost computation, potential targets are selected based on mapping of corresponding epipolar geometry. In contrast to stereo vision, the monocular epipolar flow estimation requires consecutive images containing the so-called static Focus of Expansion(FOE). Therefore, points are capable of flowing forward or backward depending on the camera’s movement. The fundamental matrix is preliminary, which allows image-level mapping in between. 

The implementation posted here is much for understanding sharing and education purpose. All work is from my interest of computer vision and without carefully refactoring.

## Basic Knowledge
As opposed to disparity estimation by stereo pairs at a time step, a flow field requires consecutive frames. More specifically, the use of the implicit epipolar geometry becomes inadequate. In the most common driver assistance system, a camera sharing the ego-motion is mounted. As a result, the new constraint called Focus of Expansion is introduced to flow images. The FOE point is the point/intersection to which all velocities of images can be extended if objects are stationary. For example from t to t+1, the FOE point is coincident. This constraint provides a potential of matching objects along extended lines and reducing the complexity. Therefore, the computational effort is similar to stereo cases. Under this setting, flow estimation holds the same data structure as disparity estimation.

The first step is to compute the fundamental matrix. The most common way is to find non-dense features existing in both images. Scale-invariant feature transform(SIFT) is an algorithm recommended for robust feature detection. Here, matcher
FlannBasedMatcher is selected for matching sparse features. RANSAC is used to improve the general robustness of the computation. After computing the fundamental matrix, the epipolar lines are estimated. In contrast to only translation, a rotation of a single camera does take place and hard to avoid. Computing rotation is required to align epipolar lines of two images. Therefore, points lying in the same epiplar lines can be matched. Afterward, using the cost function and aggregations to find the possible candidate is implemented just similar to stereo estimation.

The central body identical to the stereo case has two parts: the cost function, and aggregations. As pointed out, the aggregations in each direction are reused. Hence, adapting the cost function to epipolar flow estimation with explicit epipolar geometry becomes an important task. The other issue is spacing, which describes the distance between matched targets. In stereo estimation, the distance between two targets is constant, namely a pixel. In contrast, the spacing
in flow will be non-constant, which should be calculated. To solve this problem, Yamaguchi[1] proposed a new flow model configuring displacements which are mapped. The model must be derived for implementation. The reader could check the implementation and the paper for a better feeling. 

## Dependencies
This implementation depends on OpenCV library and extended for feature detections.

## Building
```sh
mkdir build
cd build
cmake ..
make -j
```

## Reference
[1]https://www.cs.toronto.edu/~urtasun/publications/yamaguchi_et_al_cvpr13.pdf

