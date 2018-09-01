clear
 
% Network Parameters
I  = 289;  % Input  Neurons + 1
ON = 10;   % Output Neurons
H  = 7;   % Hidden Neurons + 1
HN = H-1;  % Hidden Neurons
cc = 150;  % No of cycles
pp = 500;  % No of Patterns
l = 0.05;  % Learning Constant

% Random Initial Weights 
v = random('Normal', 0, 0.1, HN, I); % Hidden
w = random('Normal', 0, 0.1, ON, H); % Real

minErV = 0; % Initial Minimum Validation Cycle Error
 
% Initialize datasets as local variables
AC;
DC;
ground;  
bat;
ca;
am;
dio;
wire;
res;
in;

% Target Output Vectors
d1  = [1 -1 -1 -1 -1 -1 -1 -1 -1 -1] ;
d2  = [-1 1 -1 -1 -1 -1 -1 -1 -1 -1] ;
d3  = [-1 -1 1 -1 -1 -1 -1 -1 -1 -1] ;
d4  = [-1 -1 -1 1 -1 -1 -1 -1 -1 -1] ;
d5  = [-1 -1 -1 -1 1 -1 -1 -1 -1 -1] ;
d6  = [-1 -1 -1 -1 -1 1 -1 -1 -1 -1] ;
d7  = [-1 -1 -1 -1 -1 -1 1 -1 -1 -1] ;
d8  = [-1 -1 -1 -1 -1 -1 -1 1 -1 -1] ;
d9  = [-1 -1 -1 -1 -1 -1 -1 -1 1 -1] ;
d10 = [-1 -1 -1 -1 -1 -1 -1 -1 -1 1] ;
 
h1 = 0;
h2 = 50; 
for sm=1:10:500
% Recycling output matrices for each pattern

    d{sm}   = d1; 
    d{sm+1} = d2; 
    d{sm+2} = d3;
    d{sm+3} = d4;
    d{sm+4} = d5;
    d{sm+5} = d6;
    d{sm+6} = d7;
    d{sm+7} = d8;
    d{sm+8} = d9;
    d{sm+9} = d10;
    
% Recycling patterns 1-50 for training

    h1 = h1 + 1;
    
    x{sm}   = ac(:, h1);
    x{sm+1} = dc(:, h1);
    x{sm+2} = g(:, h1);
    x{sm+3} = batt(:, h1);
    x{sm+4} = cap(:, h1);
    x{sm+5} = amp(:, h1);
    x{sm+6} = diode(:, h1);
    x{sm+7} = wi(:, h1);
    x{sm+8} = resistor(:, h1);
    x{sm+9} = ind(:, h1);
end
 
% Recycling patterns 51-100 for validation
for sm=1:10:500
 
    h2 = h2 + 1;
      
    xx{sm}     = ac(:, h2);
    xx{sm+1}   = dc(:, h2);
    xx{sm+2}   = g(:, h2);
    xx{sm+3}   = batt(:, h2);
    xx{sm+4}   = cap(:, h2);
    xx{sm+5}   = amp(:, h2);
    xx{sm+6}   = diode(:, h2);
    xx{sm+7}   = wi(:, h2);
    xx{sm+8}   = resistor(:, h2);
    xx{sm+9}   = ind(:, h2);
        
end    

% Cycle testing
for c=1:cc 
    ec  = 0 ;
    ecv = 0 ;
    
% p : No of patterns
    for p=1:pp    
    x{p}(I)  = -1; % Augmentation of Input
    xx{p}(I) = -1;
    
        % Calculate Hidden Outputs
        for j=1:HN
            net   = v(j,:) * x{p};
            y(j)  = (1-exp(-net))/(1+exp(-net));
        
            net   = v(j,:) * xx{p} ;
            yy(j) = (1-exp(-net))/(1+exp(-net));    
        end
    
    % Augmentation of Hidden Outputs
    y(H)  = -1 ; % 32nd row 
    yy(H) = -1 ;
    
    % Initial Pattern Error
    EP  = 0 ;
    EPV = 0 ;
        
    for k=1:ON
        
        net = w(k,:) * y' ;
        % Output vectors
        z(k) = (1-exp(-net))/(1+exp(-net));
        
        % Delta learning rule
        outputDelta(k) = 0.5*((d{p}(k)-z(k))*(1-(z(k)^2)));

        % Pattern error calculation
        EP = EP + 0.5 * (d{p}(k) - z(k))^2;
        
        net   = w(k,:) * yy' ;
        zz(k) = (1-exp(-net))/(1+exp(-net)); % 
        EPV   = EPV + 0.5 * (d{p}(k) - zz(k))^2;
        
        % Update weights based on error
        for cm=1:H
            w2(k,cm) = w(k,cm) + l * outputDelta(k)* y(cm);
        end      
    end
    
% Hidden Delta Learning Rule
    for j=1:HN
        
        f = 0.5 * (1-y(j)^2);
        net = outputDelta * w(:,j) ;
        hiddenDelta(j)= net * f;
        
        for cm=1:I
        v2(j,cm) = v(j,cm) + l * hiddenDelta(j) * x{p}(cm);
        end   
    end
    
    v = v2 ;
    w = w2 ;
    
    % Calculate Cycle Error
    ec  = ec  + EP  ;        
    ecv = ecv + EPV ;
    
    end
    
    EC(c)  = ec  ;
    ECV(c) = ecv ;
    
    if c == 1
        minErV = ecv ;
        finalCycleV = 1 ;
        vv = v;
        ww = w;      
    % Update outputs based on minimum error
    elseif ecv < minErV 
        minErV = ecv ;
        finalCycleV = c ;
        vv = v;
        ww = w;
    end
end 
    
% Plotting the cycle error curves

minErT = min(EC) ;
finalCycleT = find(EC == min(EC)) ;
         
disp('Final Training Cycle');
disp(finalCycleT);
disp('Minimum Training Cycle Error');
disp(minErT);
disp('Final Validation Cycle');
disp(finalCycleV);
disp('Minimum Validation Cycle Error');
disp(minErV);  
 
c=1:cc;
 
f1 = figure ;
set(f1,'name','Cycle Error','numbertitle','off');
plot(c,EC,'-',c,ECV,'--') ;
xlabel('Cycles');
ylabel('Cycle Error');
legend('Training','Validation');

% Training Accuracy
 
SmS = 1;     % First Sample (Testing)
SmE = 50;    % Last  Sample (Testing)
    
% AC Power Supply 
acCtTrain = 0;
        
for sm=SmS:SmE          
    xx    = ac(:,sm); 
    xx(I) = -1;   
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;    
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end

    % Find which element is the largest
    index = find(z == max(z));
    
    if index == 1
        acCtTrain = acCtTrain + 1;  
    end
end
    
% DC Power Supply
dcCtTrain = 0;
        
for sm=SmS:SmE         
    x    = dc(:,sm);
    x(I) = -1;   
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 2
        dcCtTrain = dcCtTrain + 1;  
    end
end
    
% Ground
gCtTrain = 0 ;
        
for sm=SmS:SmE          
    xx    = g(:,sm) ;
    xx(I) = -1 ;   
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 3
      gCtTrain = gCtTrain + 1 ;  
    end
end
    
% Battery
battCtTrain = 0 ;
        
for sm=SmS:SmE   
    xx    = batt(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;  
    
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 4
      battCtTrain = battCtTrain + 1 ;  
    end
end
    
% Capacitor
capCtTrain = 0 ;
        
for sm=SmS:SmE     
    xx    = cap(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 5
      capCtTrain = capCtTrain + 1 ;  
    end
end
    
% Amplifier
ampCtTrain = 0 ;
        
for sm=SmS:SmE   
    xx    = amp(:,sm); 
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)); 
   
    if index == 6
      ampCtTrain = ampCtTrain + 1 ;  
    end
end
    
% Diode
dioCtTrain = 0 ;
        
for sm=SmS:SmE 
    xx    = diode(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 7
      dioCtTrain = dioCtTrain + 1 ;  
    end
end  
    
% Wire
wiCtTrain = 0 ;
        
for sm=SmS:SmE 
    xx    = wi(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 8
      wiCtTrain = wiCtTrain + 1 ;  
    end
end  
    
% Resistor
resCtTrain = 0 ;
        
for sm=SmS:SmE   
    xx    = resistor(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 9
      resCtTrain = resCtTrain + 1 ;  
    end
end   
    
% Inductor
indCtTrain = 0 ;
        
for sm=SmS:SmE 
    xx    = ind(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 10
      indCtTrain = indCtTrain + 1 ;  
    end
end   
     
% Accuracy as percentages

acAyTrain   = (acCtTrain   / 50) * 100;
dcAyTrain   = (dcCtTrain   / 50) * 100;
gAyTrain    = (gCtTrain    / 50) * 100;
battAyTrain = (battCtTrain / 50) * 100;
capAyTrain  = (capCtTrain  / 50) * 100;
ampAyTrain  = (ampCtTrain  / 50) * 100;
dioAyTrain  = (dioCtTrain  / 50) * 100;
wiAyTrain   = (wiCtTrain   / 50) * 100;
resAyTrain  = (resCtTrain  / 50) * 100;
indAyTrain  = (indCtTrain  / 50) * 100;
    
accuracyTrain = (acAyTrain+dcAyTrain+gAyTrain+battAyTrain+capAyTrain+ampAyTrain+dioAyTrain+wiAyTrain+resAyTrain+indAyTrain)/10 ;
       
disp('Training Accuracy');
disp(accuracyTrain);
 
% Validation Accuracy
 
SmS = 51;     % First Sample (Testing)
SmE = 100;     % Last  Sample (Testing)

acCtValidation = 0 ; % AC Power Supply 
        
for sm=SmS:SmE 
    xx    = ac(:,sm); 
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 1
      acCtValidation = acCtValidation + 1 ;  
    end
end
    
dcCtValidation = 0; % DC Power Supply
        
for sm=SmS:SmE 
    xx    = dc(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 2
      dcCtValidation = dcCtValidation + 1 ;  
    end
end
    
gCtValidation = 0 ; % Ground
        
for sm=SmS:SmE   
    xx    = g(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
            
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 3
      gCtValidation = gCtValidation + 1 ;  
    end
    end

battCtValidation = 0 ; % Battery

for sm=SmS:SmE 
    xx    = batt(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 4
      battCtValidation = battCtValidation + 1 ;  
    end
end
    
capCtValidation = 0 ; % Capacitor
        
for sm=SmS:SmE    
    xx    = cap(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 5
      capCtValidation = capCtValidation + 1 ;  
    end
end
    
ampCtValidation = 0 ; % Amplifier
        
for sm=SmS:SmE    
    xx    = amp(:,sm); 
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)); 
   
    if index == 6
      ampCtValidation = ampCtValidation + 1 ;  
    end
end

dioCtValidation = 0 ; % Diode
        
for sm=SmS:SmE   
    xx    = diode(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 7
      dioCtValidation = dioCtValidation + 1 ;  
    end
end  
    
wiCtValidation = 0 ; % Wire
        
for sm=SmS:SmE   
    xx    = wi(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 8
      wiCtValidation = wiCtValidation + 1 ;  
    end
end  

resCtValidation = 0 ; % Resistor
        
for sm=SmS:SmE   
    xx    = resistor(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 9
      resCtValidation = resCtValidation + 1 ;  
    end
end   
    
indCtValidation = 0; % Inductor
        
for sm=SmS:SmE 
    xx    = ind(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 10
        indCtValidation = indCtValidation + 1 ;  
    end
end   
    
acAyValidation   = (acCtValidation   / 50) * 100;
dcAyValidation   = (dcCtValidation   / 50) * 100;
gAyValidation    = (gCtValidation    / 50) * 100;
battAyValidation = (battCtValidation / 50) * 100;
capAyValidation  = (capCtValidation  / 50) * 100;
ampAyValidation  = (ampCtValidation  / 50) * 100;
dioAyValidation  = (dioCtValidation  / 50) * 100;
wiAyValidation   = (wiCtValidation   / 50) * 100;
resAyValidation  = (resCtValidation  / 50) * 100;
indAyValidation  = (indCtValidation  / 50) * 100;
    
accuracyValidation = (acAyValidation+dcAyValidation+gAyValidation+battAyValidation+capAyValidation+ampAyValidation+dioAyValidation+wiAyValidation+resAyValidation+indAyValidation)/10 ;
    
disp('Validation Accuracy');
disp(accuracyValidation);

% Offline Testing
    
SmS = 101; % First Sample (Testing)
SmE = 150; % Last  Sample (Testing)
acCt = 0;  % AC Power Supply
        
for sm=SmS:SmE 
    xx    = ac(:,sm); 
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 1
      acCt = acCt + 1;  
    end
end
    
dcCt = 0 ; % DC Power Supply
        
for sm=SmS:SmE 
    xx    = dc(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 2
      dcCt = dcCt + 1;  
    end
end
    
gCt = 0; % Ground
        
for sm=SmS:SmE 
    xx    = g(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
            
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 3
      gCt = gCt + 1;  
    end
end
    
battCt = 0; % Battery
        
for sm=SmS:SmE 
    xx    = batt(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 4
      battCt = battCt + 1;  
    end
end
    
capCt = 0; % Capacitor
        
for sm=SmS:SmE 
    xx    = cap(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 5
      capCt = capCt + 1;  
    end
end
    
ampCt = 0 ; % Amplifier
        
for sm=SmS:SmE 
    xx    = amp(:,sm); 
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)); 
   
    if index == 6
      ampCt = ampCt + 1 ;  
    end
end
    
dioCt = 0 ; % Diode
        
for sm=SmS:SmE 
    xx    = diode(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 7
      dioCt = dioCt + 1 ;  
    end
end  
    
wiCt = 0 ; % Wire
        
for sm=SmS:SmE   
    xx    = wi(:,sm) ;
    xx(I) = -1 ;
    for j=1:HN
        net = vv(j,:) * xx ;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 8
      wiCt = wiCt + 1 ;  
    end
end  
    
resCt = 0 ; % Resistor
        
for sm=SmS:SmE   
    xx    = resistor(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y';
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z));
    
    if index == 9
      resCt = resCt + 1;  
    end
end   
    
indCt = 0; % Inductor
        
for sm=SmS:SmE 
    xx    = ind(:,sm);
    xx(I) = -1;
    for j=1:HN
        net = vv(j,:) * xx;
        y(j) = (1-exp(-net))/(1+exp(-net));
    end
    
    y(H) = -1;
        
    for k=1:ON
        net = ww(k,:) * y' ;
        z(k) = (1-exp(-net))/(1+exp(-net));
    end
    
    index = find(z == max(z)) ;
    
    if index == 10
      indCt = indCt + 1 ;  
    end
end   
    
acAy   = (acCt   / 50) * 100;
dcAy   = (dcCt   / 50) * 100;
gAy    = (gCt    / 50) * 100;
battAy = (battCt / 50) * 100;
capAy  = (capCt  / 50) * 100;
ampAy  = (ampCt  / 50) * 100;
dioAy  = (dioCt  / 50) * 100;
wiAy   = (wiCt   / 50) * 100;
resAy  = (resCt  / 50) * 100;
indAy  = (indCt  / 50) * 100;
    
accuracy = (acAy+dcAy+gAy+battAy+capAy+ampAy+dioAy+wiAy+resAy+indAy)/10;     
disp('Testing Accuracy');
disp(accuracy);
