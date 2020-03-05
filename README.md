# Obstruction free photography
This project is taken from: https://github.com/binbin-xu/obstructionfree

## An implementation of the paper "A Computational Approach for Obstruction-Free Photography

GSoC OpenCV 2017 prject:

Implementation of the SIGGRAPH 2015 paper "A Computational Approach for Obstruction-Free Photography" (https://sites.google.com/site/obstructionfreephotography/).

[GSoC Project Page is here](https://summerofcode.withgoogle.com/dashboard/project/6685947065270272/overview/)

[Original project proposal is here](https://storage.googleapis.com/summerofcode-prod.appspot.com/gsoc/core_project/doc/4838076040871936_1491143437_ComputationalOcclusionRemovalinImageInpainting--GSoC2017proposalforopenCV_2.pdf?Expires=1496766357&GoogleAccessId=summerofcode-prod%40appspot.gserviceaccount.com&Signature=TCdVHQWJgpI%2FHS8ijzLdFsyJnXudrrisypDbk%2BolVbwpWIJ3I7So%2BR%2B5RPVFe7VKT5N%2FE1iGFp7gm7vtM0f%2BkLBmh1%2BRjE2QfyeX3kNdzI6NhcByd4hDoaLn8FS%2BuqgWyjrsQ%2BFvCfO0UQdbREsOavagPaw9EYdPHaR%2F7lCjFlMMwKhyKGGaLXBTvRTT3lazklCOJFmcfMZe2nfNVscbPLEs9ZseGjKIYOiQ85le0%2BgWTE%2BsJdv6ueTquJ%2BcHv6VY03GfjQMPxIzSQSwFAHTfl%2BA7zwxmKNEjWXSabVMNmudYz0MHcuAZmS%2FaRSp%2BvsMPM7926V8tYLwCiUegLABbA%3D%3D)

Considering to fix if needed:
1. To fix the Matrix data type in initial decomostion
2. To fix image warping in the case of "outlier" correspondence in the borders
3. edgeflow part to MRF framework 
4. visual surface interpolation, currently using epicflow interpolation

To add if needed:
1. swtich image warping from direct alignment to iterative warping
2. Add a parameter to choose larger motion fields as foreground or backgrond motion layers

On the scheduel, next:
1. Layer decompostion-solving Eq.10 using IRLS
2. Update motion estimation -solving Eq.14 using IRLS

Binbin Xu

## Requirements
		- OpenCV 4.1.2
		- CMake 3.5.1 minimum		


## Compilation

		- mkdir build
		- cd build/
		- cmake ../
		- make

## Test

		- cd build/
		- ./obstructionfre outcpp.avi
