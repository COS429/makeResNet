function makeResNet() 
    fileID = fopen('ResNet.json','w');

    % Print header
    path = 'examples/imagenet/';
    snapshot_prefix = 'examples/imagenet/alexnet_imagenet_snapshot';
    solver = 'SGD';
    regularizer = 'L2';
    momentum = 0.9;
    weight_decay = 0.0005;
    base_lr = 0.01;
    lr_policy = 'LR_step';
    lr_gamma = 0.1;
    lr_stepsize = 100000;
    max_iter = 450000;
    snapshot_iter = 10000;
    display_iter = 20;
    test_iter = 100;
    test_interval = 100;
    fprintf(fileID,'{\n\t"train":{\n\t\t"path":"%s",\n\t\t"snapshot_prefix":"%s",\n\t\t"solver":"%s",\n\t\t"regularizer":"%s",\n\t\t"momentum":%f,\n\t\t"weight_decay":%f,\n\t\t"base_lr":%f,\n\t\t"lr_policy":"%s",\n\t\t"lr_gamma":%f,\n\t\t"lr_stepsize":%d,\n\t\t"max_iter":%d,\n\t\t"snapshot_iter":%d,\n\t\t"display_iter":%d,\n\t\t"test_iter":%d,\n\t\t"test_interval":%d,\n\t\t"GPU":[0],\n\t\t"debug_mode":false\n\t},\n\t"test":{\n\t\t"GPU":0,\n\t\t"debug_mode":false\n\t},\n\t"layers":[\n',path,snapshot_prefix,solver,regularizer,momentum,weight_decay,base_lr,lr_policy,lr_gamma,lr_stepsize,max_iter,snapshot_iter,display_iter,test_iter,test_interval);

    % Print data layer

%     res_id = 1;
%     conv1_num_output = 64;
%     conv1_window = 3;
%     conv2_num_output = 64;
%     conv2_window = 3;
%     is_shortcut = true;


    fprintf(fileID,'\n\t\t{\n\t\t\t"bias_decay_mult":0.0,\n\t\t\t"bias_filler":"Constant",\n\t\t\t"bias_filler_param":0.0,\n\t\t\t"bias_lr_mult":2.0,\n\t\t\t"in":[\n\t\t\t\t"data"\n\t\t\t],\n\t\t\t"name":"conv1",\n\t\t\t"num_output":64,\n\t\t\t"out":[\n\t\t\t\t"conv1"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t2,\n\t\t\t\t2\n\t\t\t],\n\t\t\t"type":"Convolution",\n\t\t\t"weight_decay_mult":1.0,\n\t\t\t"weight_filler":"Gaussian",\n\t\t\t"weight_filler_param":0,\n\t\t\t"weight_lr_mult":1.0,\n\t\t\t"window":[\n\t\t\t\t7,\n\t\t\t\t7\n\t\t\t]\n\t\t},');

    fprintf(fileID,'\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"conv1"\n\t\t\t],\n\t\t\t"mode":"ReLU",\n\t\t\t"name":"relu1",\n\t\t\t"out":[\n\t\t\t\t"conv1"\n\t\t\t],\n\t\t\t"type":"Activation"\n\t\t},');

    fprintf(fileID,'\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"conv1"\n\t\t\t],\n\t\t\t"mode":"max",\n\t\t\t"name":"pool1",\n\t\t\t"out":[\n\t\t\t\t"pool1"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t2,\n\t\t\t\t2\n\t\t\t],\n\t\t\t"type":"Pooling",\n\t\t\t"window":[\n\t\t\t\t2,\n\t\t\t\t2\n\t\t\t]\n\t\t},');

    print_residual_block(fileID, 1, 64, 3, 64, 3, 'pool1', 'res1out', false);
    print_residual_block(fileID, 2, 64, 3, 64, 3, 'res1out', 'res2out', false);
    print_residual_block(fileID, 3, 64, 3, 64, 3, 'res2out', 'res3out', false);
    print_residual_block(fileID, 4, 128, 3, 128, 3, 'res3out', 'res4out', true);
    print_residual_block(fileID, 5, 128, 3, 128, 3, 'res4out', 'res5out', false);
    print_residual_block(fileID, 6, 128, 3, 128, 3, 'res5out', 'res6out', false);
    print_residual_block(fileID, 7, 128, 3, 128, 3, 'res6out', 'res7out', false);
    print_residual_block(fileID, 8, 256, 3, 256, 3, 'res7out', 'res8out', true);
    print_residual_block(fileID, 9, 256, 3, 256, 3, 'res8out', 'res9out', false);
    print_residual_block(fileID, 10, 256, 3, 256, 3, 'res9out', 'res10out', false);
    print_residual_block(fileID, 11, 256, 3, 256, 3, 'res10out', 'res11out', false);
    print_residual_block(fileID, 12, 256, 3, 256, 3, 'res11out', 'res12out', false);
    print_residual_block(fileID, 13, 256, 3, 256, 3, 'res12out', 'res13out', false);
    print_residual_block(fileID, 14, 512, 3, 512, 3, 'res13out', 'res14out', true);
    print_residual_block(fileID, 15, 512, 3, 512, 3, 'res14out', 'res15out', false);
    print_residual_block(fileID, 16, 512, 3, 512, 3, 'res15out', 'res16out', false);



    fprintf(fileID,'\n\t]\n}');
    fclose(fileID);

end



function print_residual_block(fileID, res_id, conv1_num_output, conv1_window, conv2_num_output, conv2_window, res_block_input, res_block_output, is_shortcut)
%MAKE Summary of this function goes here
%   Detailed explanation goes here


%     res_block_input = sprintf('res%din',res_id);
    conv1_name = sprintf('res%dconv1',res_id);
    relu1_name = sprintf('res%drelu1',res_id);

    conv2_name = sprintf('res%dconv2',res_id);
    relu2_name = sprintf('res%drelu2',res_id);
%     res_block_output = sprintf('res%dout',res_id);

    elementwise_name = sprintf('ElementWise%d',res_id);
    res_block_inter = sprintf('res%dinter',res_id);

    if is_shortcut
        % 3x3 Convolution + /2 Stride + ReLu
        fprintf(fileID, '\n\t\t{\n\t\t\t"bias_decay_mult":0.0,\n\t\t\t"bias_filler":"Constant",\n\t\t\t"bias_filler_param":0.0,\n\t\t\t"bias_lr_mult":2.0,\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"name":"%s",\n\t\t\t"num_output":%d,\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t2,\n\t\t\t\t2\n\t\t\t],\n\t\t\t"type":"Convolution",\n\t\t\t"weight_decay_mult":1.0,\n\t\t\t"weight_filler":"Gaussian",\n\t\t\t"weight_filler_param":0,\n\t\t\t"weight_lr_mult":1.0,\n\t\t\t"window":[\n\t\t\t\t%d,\n\t\t\t\t%d\n\t\t\t]\n\t\t},\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ReLU",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"Activation"\n\t\t},',res_block_input,conv1_name,conv1_num_output,conv1_name,conv1_window,conv1_window,conv1_name,relu1_name,conv1_name);
    else
        % 3x3 Convolution + ReLu
        fprintf(fileID, '\n\t\t{\n\t\t\t"bias_decay_mult":0.0,\n\t\t\t"bias_filler":"Constant",\n\t\t\t"bias_filler_param":0.0,\n\t\t\t"bias_lr_mult":2.0,\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"name":"%s",\n\t\t\t"num_output":%d,\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"type":"Convolution",\n\t\t\t"weight_decay_mult":1.0,\n\t\t\t"weight_filler":"Gaussian",\n\t\t\t"weight_filler_param":0,\n\t\t\t"weight_lr_mult":1.0,\n\t\t\t"window":[\n\t\t\t\t%d,\n\t\t\t\t%d\n\t\t\t]\n\t\t},\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ReLU",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"Activation"\n\t\t},',res_block_input,conv1_name,conv1_num_output,conv1_name,conv1_window,conv1_window,conv1_name,relu1_name,conv1_name);
    end
        
    % 3x3 Convolution + ReLu
    fprintf(fileID, '\n\t\t{\n\t\t\t"bias_decay_mult":0.0,\n\t\t\t"bias_filler":"Constant",\n\t\t\t"bias_filler_param":0.0,\n\t\t\t"bias_lr_mult":2.0,\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"name":"%s",\n\t\t\t"num_output":%d,\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"type":"Convolution",\n\t\t\t"weight_decay_mult":1.0,\n\t\t\t"weight_filler":"Gaussian",\n\t\t\t"weight_filler_param":0,\n\t\t\t"weight_lr_mult":1.0,\n\t\t\t"window":[\n\t\t\t\t%d,\n\t\t\t\t%d\n\t\t\t]\n\t\t},',conv1_name,conv2_name,conv2_num_output,conv2_name,conv2_window,conv2_window);
    if is_shortcut
        % 1x1 Convolution
        fprintf(fileID, '\n\t\t{\n\t\t\t"bias_decay_mult":0.0,\n\t\t\t"bias_filler":"Constant",\n\t\t\t"bias_filler_param":0.0,\n\t\t\t"bias_lr_mult":2.0,\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"name":"%s",\n\t\t\t"num_output":%d,\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"padding":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"stride":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t],\n\t\t\t"type":"Convolution",\n\t\t\t"weight_decay_mult":1.0,\n\t\t\t"weight_filler":"Gaussian",\n\t\t\t"weight_filler_param":0,\n\t\t\t"weight_lr_mult":1.0,\n\t\t\t"window":[\n\t\t\t\t1,\n\t\t\t\t1\n\t\t\t]\n\t\t},', res_block_input, res_block_inter, conv1_num_output, res_block_inter);
        % Elementwise Sum
        fprintf(fileID, '\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s",\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ElementWise_SUM",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"ElementWise"\n\t\t},', conv2_name, res_block_inter, elementwise_name, res_block_output);
        % ReLu
        fprintf(fileID, '\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ReLU",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"Activation"\n\t\t},',res_block_output,relu2_name,res_block_output);
    else 
        % Elementwise Sum
        fprintf(fileID, '\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s",\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ElementWise_SUM",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"ElementWise"\n\t\t},',res_block_input,conv2_name,elementwise_name,res_block_output);
        % ReLu
        fprintf(fileID, '\n\t\t{\n\t\t\t"in":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"mode":"ReLU",\n\t\t\t"name":"%s",\n\t\t\t"out":[\n\t\t\t\t"%s"\n\t\t\t],\n\t\t\t"type":"Activation"\n\t\t},',res_block_output,relu2_name,res_block_output);
    end

end

