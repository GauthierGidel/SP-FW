function h = hashing(sequence)
    % sequence is vector of -1 and 1 (for unit box)
    % if any fractional is present, just map to something
    % output is string with a for -1, b for 1 and c for fractional
    if any(abs(sequence.^2 - 1) > eps)
        % some fractional corner:
        h = 'c';
    else
        h = repmat('a',1,length(sequence));
        h(sequence == 1) = 'b';
    end
    
end


