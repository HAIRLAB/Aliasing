function [output] = make_output_fd(Y)

   dim = size(Y,1);
    
    name_num = 1;

    for j = 1:dim
        name_tmp = [strcat('y(',num2str(j)),')'];
        name{name_num} = name_tmp;    
        name_num = name_num+1;
    end
    output.name = name;
    output.variable = Y';

    
end
