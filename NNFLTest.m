clear

NNFLProject;
res1     ; 
    
xx    = a ; % Temporary matrix
xx(I) = -1; % Augmentation
    
for j=1:HN
    net = vv(j,:) * xx ;
    yy(j) = (1-exp(-net))/(1+exp(-net));
end
    
yy(H) = -1; % Augmentation
        
for k=1:ON
    net = ww(k,:) * yy';
    zz(k) = (1-exp(-net))/(1+exp(-net))
end

% Find the element closest to 1
index = find(zz == max(zz)); 
switch index
    case 1
        disp('AC');
    case 2
        disp('DC');
    case 3
        disp('Ground');
    case 4
        disp('Battery');
    case 5
        disp('Capacitor');
    case 6
        disp('Amplifier');
    case 7
        disp('Diode');
    case 8
        disp('Wire');
    case 9
        disp('Resistor');
    case 10
        disp('Inductor');
end
