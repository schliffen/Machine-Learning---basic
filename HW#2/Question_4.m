clear; clc; close all

addpath('/Users/alisk/Desktop/340 HW/a2');
addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW\340 HW\a2');

load dog.mat

% image(I/256)
figure
[Iquant] = quantizeImage(I,1);
imshow(Iquant/256);
title('(Question 4.1.2) k=2^1')

figure
[Iquant] = quantizeImage(I,2);
imshow(Iquant/256);
title('(Question 4.1.2) k=2^2')

figure
[Iquant] = quantizeImage(I,4);
imshow(Iquant/256);
title('(Question 4.1.2) k=2^4')

figure
[Iquant] = quantizeImage(I,6);
imshow(Iquant/256);
title('(Question 4.1.2) k=2^6')