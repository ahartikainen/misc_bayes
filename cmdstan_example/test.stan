parameters {
    real y;
    vector[3] x;
    matrix[4,6] Z;
}
model {
    y ~ normal(0,1);
    for (n in 1:3) {
        x[n] ~ normal(n, 1);
    }
    for (n in 1:4) {
        for (m in 1:6) {
	    Z[n, m] ~ normal(n, m);
	}
    }
}
