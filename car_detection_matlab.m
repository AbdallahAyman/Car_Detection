%foreground detection training
foregroundDetector = vision.ForegroundDetector('NumGaussians', 5, ...
    'NumTrainingFrames', 50);
videoReader = vision.VideoFileReader('C:\Users\bedo__000\Desktop\GUC\Image processing\BMY\LB2.webm');
for i = 1:150
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end
%blob analysis
blob = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 1500);
%initiate videoplayer
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [800,400];  % window size: [width, height]
se = strel('square', 3); % morphological filter for noise removal

%video playing loop
while ~isDone(videoReader)
    
    
    frame = step(videoReader); % read the next video frame
    
    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);
    
    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);
    
    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    [Area,blobbox] = step(blob, filteredForeground);
    %initialize some parameter that will be used later
    dummy=blobbox;
    blobyellow=[];
    ny=[];
    if isempty(blobbox)==0
        n=size(blobbox(:,1));
        %loop for cropping each blob from blobbox
        for i=1:n(1)
            %crop blob
            x1=blobbox(i,1);%top left corner x coordinate
            y1=blobbox(i,2);%top left corner y coordinate
            x2=blobbox(i,3);%width from x
            y2=blobbox(i,4);%height from y
            I2 = imcrop(frame,[x1 y1 x2 y2]);
            %detect blobs of yellow color
            Im= color_detection_by_hue(I2);
            X=Im.yellow;%black and white image where black is other colours and white is yellow
            [counts,binLocations] = imhist(X);
            nones=counts(2,:);%no.of white pixels
            nzeroes=counts(1,:);%no. of black pixels
            ntotal=nones+nzeroes;%total no. of pixels
            ratio=nones/ntotal;%ratio of white to black
            
            if ratio>0.25 %threshold for ratio
                blobyellow=[blobyellow;blobbox(i,:)];%put yellow blobs in a matrix
                ny=[ny;i];%array containing indeces of yellow blobs in the original blobox
                
            else
                %continue with original frame and blobbox if no yellow blobs found
                result=frame;
                blobbox2=blobbox;
            end
            %insert a rectangle around yellow cars
            result = insertShape(frame, 'Rectangle', blobyellow, 'Color', 'yellow');
        end
        %remove rows containing yellow blobs from original blobbox(dummy)
        %we updtate the indeces of the remaining yellow blobs when one is removed     
        if isempty(ny)==0
            for j=1:size(ny(:,1))
                blobbox2=removerows(dummy,ny(j,:));
                dummy=blobbox2;
                for k=1:size(ny(:,1))
                    ny(k,:)=ny(k,:)-1;
                end
            end
        end
        %insert rectangle around non-yellow cars
        result=insertShape(result, 'Rectangle', blobbox2, 'Color', 'green');
        % Display the number of non-yellow cars and yellow cars found in the
        % video frame in 2 separate boxes
        numCars = size(blobbox2, 1);
        result = insertText(result, [10 10], numCars,'BoxColor','green', 'BoxOpacity', 1, ...
            'FontSize', 14);
        numyCars = size(blobyellow, 1);
        result = insertText(result, [50 10], numyCars,'BoxColor','yellow', 'BoxOpacity', 1, ...
            'FontSize', 14);
        %display the result frame on the video player
        step(videoPlayer, result);
        
        
        
        
        
   
    else
        %if no cars found, print 2 zeroes in the 2 boxes
        numCars=0;
        frame = insertText(frame, [10 10], numCars,'BoxColor','green', 'BoxOpacity', 1, ...
            'FontSize', 14);
        numyCars=0;
        frame = insertText(frame, [50 10], numyCars,'BoxColor','yellow', 'BoxOpacity', 1, ...
            'FontSize', 14);
        %output the frame
        step(videoPlayer, frame);
        
    end
end
release(videoReader);% close the video file