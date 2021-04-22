%Resshapes the 1x784 image vector into a viewable picture 28x28

function imagedrawer(imagev, tag)
    if isequal(size(imagev), [1,784])
        imagev = reshape(imagev, [28,28]);
    end
    
    imagev = fliplr(imagev);
    imagev = rot90(imagev);
    
    image(imagev);
    if nargin == 2
        title(sprintf('%d', tag));
    end
end