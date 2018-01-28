%{ @file make_gif.m
%  @brief Matlab script that plots the results of diffusion.c/cu .
%
%  This script plots the results of the heat diffusion equation solver diffusion.c/cu and saves it
%  into a gif file.
%
%  @author Richard Finger
%}

figure
hold on

for i = 1:41

    filename = 'paralel_video.gif';
    name = strcat('matrix',num2str(i),'.csv');
    M = csvread(name);

    xlim([0 40]);
    ylim([0 40]);

    colormap(hot)
    imagesc(M);

    title(num2str(i));

    pause(0.005);

    frame = getframe(1);
    im = frame2im(frame);
    [A,map] = rgb2ind(im,256);

	if i == 1;
		imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
	else
		imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1);
	end
end
