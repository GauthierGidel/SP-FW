function id = away_step(grad, S, I_active)
% returns the id of the active atom with the worst value w.r.t. the
% gradient
    s = grad' * S(:,I_active);
    [~,id] = max(s);
    id = I_active(id(1));
end

