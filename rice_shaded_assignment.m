%Assigment by Aldo Villarreal on January 6th
%More gray level image processing

%Import image to workspace
rice=imread("rice-shaded.tif");
figure(1)
imshow(rice)

%Create filter to mask uneven ilumination
h_filter=fspecial('average',163);
%Apply filter on the original image using replicate to create the mask
rice_average=imfilter(rice,h_filter,'replicate');
figure(2)
imshow(rice_average)
%Convert the images to double to improve performance during computation
rice_average=double(rice_average)/255;
rice=double(rice)/255;

%Apply mask on the original image. 
rice_even=rice./rice_average;
figure(3)
imshow(rice_even)

%Correct image by multiplying the mean value of the original image
rice_correct=rice_even*mean(rice(:));
figure(4)
imshow(rice_correct)

%Calculate threshold by using The Otsu method
th=graythresh(rice_correct);
%Apply threshold to image after ilumination has been corrected
rice_otsu=rice_correct>th;
figure(5)
%Compare original image vs processed image
montage([rice rice_otsu])

%label all pieces of rice on the image
labels=bwlabel(rice_otsu,8);

%Compute Areas and Centroids about the binary image
stats=regionprops("table",labels,"Area","Centroid");
cent=stats.Centroid;
areas = stats.Area;

figure(6)
subplot(2,1,1)
imshow(labels)
hold on
%Plot centroids for each rice piece on top of the image
plot(cent(:,1),cent(:,2),'r.'); 
hold off

%Add histogram under the image
subplot(2,1,2)
histogram(areas)
xlabel("Area in Pixels")
ylabel("Count of Rice")