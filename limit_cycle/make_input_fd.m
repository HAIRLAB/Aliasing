function [input] = make_input_fd(X)

   dim = size(X,1);
    
    name_num = 1;

    for j = 1:dim
        name_tmp = [strcat('x(',num2str(j)),')'];
        name{name_num} = name_tmp;    
        name_num = name_num+1;
    end
    input.name = name;
    input.variable = X';

    
end
