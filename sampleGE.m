
epsilon(1,:) = 0.1:0.1:0.8;
num_frames = 10000;

% Forward GE channel
s = 0.3; %r = 0.1;
eps_set = epsilon(1,:);         %the probability of block error (design requirement)
qset = eps_set*s./(1-eps_set);  %since eps_B q/(q+s) = eps, and eps_B = 1


[ChannelStatef,~,~] = GEchannelUncodedARQ(qset(1),s,num_frames);
GE = ChannelStatef;
sum(ChannelStatef)

function [ChannelStatef,ChannelStater,Pkron] = GEchannelUncodedARQ(q,s,total_no)

    P = [1-q, q; s, 1-s];
    Pkron = kron(P,P);
    pi_G = s/(s+q); %probability of being in State G
    pi_B = q/(s+q); %probability of being in State B
   
    ChannelStatef = zeros(1,total_no);
    ChannelStater = zeros(1,total_no);
    
    goodf = rand(1) > pi_B;
    goodr = rand(1) > pi_B;          

    for size = 1:total_no
        % goodf = 1 and goodr = 1 if next step is bad (=erasure/defective)
        if goodf == 1 && goodr == 1        
            goodf = rand(1) > q ; % move to bad
            goodr = rand(1) > q ; % move to bad

        elseif goodf == 1 && goodr == 0
            goodf = rand(1) > q ; % move to bad
            goodr = rand(1) > 1-s; % stay bad

        elseif goodf == 0 && goodr == 1
            goodf = rand(1) > 1-s; % stay bad
            goodr = rand(1) > q; % move to bad

        elseif goodf == 0 && goodr == 0
            goodf = rand(1) > 1-s; % stay bad
            goodr = rand(1) > 1-s; % stay bad

        end
        ChannelStatef(size) = goodf;
        ChannelStater(size) = goodr;
    end
                
end