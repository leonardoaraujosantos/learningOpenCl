%% Measuring GPU Transfer rate
% https://uk.mathworks.com/help/matlab/ref/timeit.html
% https://uk.mathworks.com/help/distcomp/gpuarray.html
% http://blogs.mathworks.com/loren/2012/12/14/measuring-gpu-performance/
% http://blogs.mathworks.com/pick/2013/05/17/benchmarking-your-gpu/
% http://uk.mathworks.com/company/newsletters/articles/improvements-to-tic-and-toc-functions-for-measuring-absolute-elapsed-time-performance-in-matlab.html?s_cid=fb_wall_11-8-11_newsletter_tictoc
% http://uk.mathworks.com/help/distcomp/examples/benchmarking-a-b-on-the-gpu.html
% https://uk.mathworks.com/help/distcomp/measure-and-improve-gpu-performance.html

%% Setup
gpu = gpuDevice();
fprintf('Using a %s GPU.\n', gpu.Name)
sizeOfByte = 1; % Each double-precision number needs 8 bytes of storage
sizes = power(2, 1:24);

%% Testing host/GPU bandwidth
% The first test estimates how quickly data can be sent to and
% read from the GPU.  Because the GPU is plugged into the PCI bus, this
% largely depends on how fast the PCI bus is and how many other things are
% using it.  However, there are also some overheads that are included in
% the measurements, particularly the function call overhead and the array
% allocation time.  Since these are present in any "real world" use of the
% GPU, it is reasonable to include these.
%
% In the following tests, memory is allocated and data is sent to the GPU
% using the <matlab:doc('gpuArray') |gpuArray|> function.  Memory is
% allocated and data is transferred back to host memory using 
% <matlab:doc('gpuArray/gather') |gather|>.
%
% Note that PCI express v2, as used in this test, has a theoretical
% bandwidth of 0.5GB/s per lane. For the 16-lane slots (PCIe2 x16) used by
% NVIDIA's compute cards this gives a theoretical 8GB/s.
sendTimes = inf(size(sizes));
gatherTimes = inf(size(sizes));
for ii=1:numel(sizes)
    numElements = sizes(ii)/sizeOfByte;
    hostData = uint8(randi([0 9], numElements, 1));
    gpuData = uint8(randi([0 9], numElements, 1, 'gpuArray'));
    % Time sending to GPU
    sendFcn = @() gpuArray(hostData);
    sendTimes(ii) = gputimeit(sendFcn);
    % Time gathering back from GPU
    gatherFcn = @() gather(gpuData);
    gatherTimes(ii) = gputimeit(gatherFcn);
end
sendBandwidth = (sizes./sendTimes)/1e9;
[maxSendBandwidth,maxSendIdx] = max(sendBandwidth);
fprintf('Achieved peak send speed of %g GB/s\n',maxSendBandwidth)
gatherBandwidth = (sizes./gatherTimes)/1e9;
[maxGatherBandwidth,maxGatherIdx] = max(gatherBandwidth);
fprintf('Achieved peak gather speed of %g GB/s\n',max(gatherBandwidth))

%%
% On the plot below, the peak for each case is circled.  With small data
% set sizes, overheads dominate.  With larger amounts of data the PCI bus
% is the limiting factor.
hold off
semilogx(sizes, sendBandwidth, 'b.-', sizes, gatherBandwidth, 'r.-')
hold on
semilogx(sizes(maxSendIdx), maxSendBandwidth, 'bo-', 'MarkerSize', 10);
semilogx(sizes(maxGatherIdx), maxGatherBandwidth, 'ro-', 'MarkerSize', 10);
grid on
title('Data Transfer Bandwidth')
xlabel('Array size (bytes)')
ylabel('Transfer speed (GB/s)')
legend('Send to GPU', 'Gather from GPU', 'Location', 'NorthWest')


%% Testing memory intensive operations
% Many operations do very little computation with each element of an array
% and are therefore dominated by the time taken to fetch the data from
% memory or to write it back.  Functions such as |ones|, |zeros|, |nan|,
% |true| only write their output, whereas functions like |transpose|,
% |tril| both read and write but do no computation.  Even simple operators
% like |plus|, |minus|, |mtimes| do so little computation per element that
% they are bound only by the memory access speed. 
%
% The function |plus| performs one memory read and one memory write for
% each floating point operation.  It should therefore be limited by memory
% access speed and provides a good indicator of the speed of a read+write
% operation.
memoryTimesGPU = inf(size(sizes));
for ii=1:numel(sizes)
    numElements = sizes(ii)/sizeOfByte;
    gpuData = randi([0 9], numElements, 1, 'gpuArray');
    plusFcn = @() plus(gpuData, 1.0);
    memoryTimesGPU(ii) = gputimeit(plusFcn);
end
memoryBandwidthGPU = 2*(sizes./memoryTimesGPU)/1e9;
[maxBWGPU, maxBWIdxGPU] = max(memoryBandwidthGPU);
fprintf('Achieved peak read+write speed on the GPU: %g GB/s\n',maxBWGPU)

%%
% Now compare it with the same code running on the CPU.
memoryTimesHost = inf(size(sizes));
for ii=1:numel(sizes)
    numElements = sizes(ii)/sizeOfByte;
    hostData = randi([0 9], numElements, 1);
    plusFcn = @() plus(hostData, 1.0);
    memoryTimesHost(ii) = timeit(plusFcn);
end
memoryBandwidthHost = 2*(sizes./memoryTimesHost)/1e9;
[maxBWHost, maxBWIdxHost] = max(memoryBandwidthHost);
fprintf('Achieved peak read+write speed on the host: %g GB/s\n',maxBWHost)

% Plot CPU and GPU results.
figure;
hold off
semilogx(sizes, memoryBandwidthGPU, 'b.-', ...
    sizes, memoryBandwidthHost, 'r.-')
hold on
semilogx(sizes(maxBWIdxGPU), maxBWGPU, 'bo-', 'MarkerSize', 10);
semilogx(sizes(maxBWIdxHost), maxBWHost, 'ro-', 'MarkerSize', 10);
grid on
title('Read+write Bandwidth')
xlabel('Array size (bytes)')
ylabel('Speed (GB/s)')
legend('GPU', 'Host', 'Location', 'NorthWest')

