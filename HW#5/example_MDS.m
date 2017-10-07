clear
clc
close all
load animals.mat
% Z = visualizeMDS(X,2,animals);
Z = visualizeISOMAP(X,2,animals);