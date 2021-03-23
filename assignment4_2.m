% load Iris Dataset
close all hidden

load iris_dataset

x=irisInputs;

% selforgmap creates self-organizing maps for classifying samples with  as much detail as desired 
% by selecting the number of neurons in each dimension of the layer.

net = selforgmap([8 8]);

%view(net)


% Now the network is ready to be optimized with train. 
% The Neural Network Training Tool shows the network being trained and the algorithms used to train it.
% It also displays the training state during training and the criteria which stopped training will be highlighted in green. 

[net,tr] = train(net,x);

nntraintool


% Here the self-organizing map is used to compute the class vectors of each of the training inputs. 
% These classifications cover the feature space populated by the known flowers,
% and can now be used to classify new flowers accordingly. 
% The network output will be 64x150 matrix, 
% where each ith column represents the jth cluster for each ith input vector with a 1 in its jth element. 
% The function vec2ind returns the index of the neuron with an output of 1, for each vector. 
% The indices will range between 1 and 64 for the 64 clusters represented by the 64 neurons.

 y = net(x);

cluster_index = vec2ind(y);


figure, plotsomtop(net) %Plots topology of its neurons

figure, plotsomhits(net,x) %Plot the number of samples assigned to each neuron in the map (Sample hits)
%calculates the classes for each flower and shows the number of flowers in each class. 
% Areas of neurons with large numbers of hits indicate classes representing similar highly populated regions of the feature space. 
% Whereas areas with few hits indicate sparsely populated regions of the feature space.
%it's a good way to view the distribution of the data set on the map.

figure, plotsomnc(net) %Plots the connections between adjacent neurons

figure, plotsomnd(net) % displays the distances between the weight vectors of adjacent neurons in the map.
%shows how distant (in terms of Euclidian distance) each neuron's class is from its neighbors.

%The blue hexagons represent the neurons.
%The red lines connect neighboring neurons.
%The colors in the regions containing the red lines indicate the
% distances between neurons.
% The darker colors represent larger distances.
% The lighter colors represent smaller distances.

figure, plotsomplanes(net)
%Creates a plot for each input element "i", showing how strongly different
%areas of the map connect to input i negatively or positively. 

figure, plotsompos(net,x)
%plot the input data alongside the weights.