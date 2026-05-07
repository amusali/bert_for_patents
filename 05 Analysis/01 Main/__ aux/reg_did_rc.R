#@title Func replace

fnc <-  function (y, post, D, covariates, i.weights = NULL, boot = FALSE,
    boot.type = "weighted", nboot = NULL, inffunc = FALSE)
{
    D <- as.vector(D)
    post <- as.vector(post)
    n <- length(D)
    y <- as.vector(y)
    if (is.null(covariates)) {
        int.cov <- as.matrix(rep(1, n))
    }
    else {
        int.cov <- as.matrix(covariates)
    }
    if (is.null(i.weights)) {
        i.weights <- as.vector(rep(1, n))
    }
    else if (min(i.weights) < 0)
        stop("i.weights must be non-negative")
    i.weights <- i.weights/mean(i.weights)

    keep_lindep_cols <- function(X, tol = 1e-10, keep_intercept = TRUE) {
    X <- as.matrix(X)

    is_const_one <- apply(X, 2, function(z) all(is.finite(z)) && max(abs(z - 1)) < tol)
    is_nonconst <- apply(X, 2, function(z) {
        z <- z[is.finite(z)]
        if (length(z) <= 1) return(FALSE)
        stats::var(z) > tol
    })

    keep_var <- is_nonconst | (keep_intercept & is_const_one)

    if (!any(keep_var)) {
        stop("No covariate columns with variation remain in this subsample.")
    }

    X1 <- X[, keep_var, drop = FALSE]

    qx <- qr(X1, tol = tol)
    keep_qr <- qx$pivot[seq_len(qx$rank)]
    X2 <- X1[, keep_qr, drop = FALSE]

    list(
        X_reduced = X2,
        keep_var = keep_var,
        keep_qr = keep_qr,
        rank = qx$rank
    )
}

    pre_filter <- (D == 0) & (post == 0)

    cat("\n===== PRE REGRESSION DEBUG =====\n")
    cat("n total =", n, "\n")
    cat("pre_filter count (controls, pre) =", sum(pre_filter), "\n")
    cat("treated pre count =", sum((D == 1) & (post == 0)), "\n")
    cat("control post count =", sum((D == 0) & (post == 1)), "\n")
    cat("treated post count =", sum((D == 1) & (post == 1)), "\n")
    cat("dim(int.cov) =", paste(dim(int.cov), collapse = " x "), "\n")
    cat("dim(int.cov[pre_filter, ]) =", paste(dim(int.cov[pre_filter, , drop = FALSE]), collapse = " x "), "\n")
    cat("number of original regressors =", ncol(int.cov), "\n")
    cat("sum of pre weights =", sum(i.weights[pre_filter]), "\n")
    cat("================================\n")


   # reg.coeff.pre <- stats::coef(fastglm::fastglm(x = int.cov[pre_filter,
      #  , drop = FALSE], y = y[pre_filter], weights = i.weights[pre_filter],
      #  family = gaussian(link = "identity")))
    #Edit start
    x_pre_raw <- int.cov[pre_filter, , drop = FALSE]
    pre_red <- keep_lindep_cols(x_pre_raw)

    x_pre <- pre_red$X_reduced
    tmp_pre_full <- int.cov[, pre_red$keep_var, drop = FALSE]
    int.cov.pre <- tmp_pre_full[, pre_red$keep_qr, drop = FALSE]

    y_pre <- y[pre_filter]
    w_pre <- i.weights[pre_filter]

    cat("\n--- PRE COLUMN REDUCTION ---\n")
    cat("original regressors =", ncol(int.cov), "\n")
    cat("after dropping no-variance cols =", sum(pre_red$keep_var), "\n")
    cat("final pre rank =", ncol(x_pre), "\n")
    cat("----------------------------\n")


# original package regression
fit.pre <- fastglm::fastglm(
  x = x_pre,
  y = y_pre,
  weights = w_pre,
  family = gaussian(link = "identity")
)
reg.coeff.pre <- stats::coef(fit.pre)

    cat("\n===== PRE REGRESSION RESULTS =====\n")
    #print(reg.coeff.pre)
    cat("==================================\n")
    #Edit end
    if (anyNA(reg.coeff.pre)) {
        stop("Outcome regression model coefficients have NA components. \n Multicollinearity of covariates is probably the reason for it.")
    }
    out.y.pre <- as.vector(tcrossprod(reg.coeff.pre, int.cov.pre))

    post_filter <- (D == 0) & (post == 1)

x_post_raw <- int.cov[post_filter, , drop = FALSE]
post_red <- keep_lindep_cols(x_post_raw)

x_post <- post_red$X_reduced
tmp_post_full <- int.cov[, post_red$keep_var, drop = FALSE]
int.cov.post <- tmp_post_full[, post_red$keep_qr, drop = FALSE]

cat("\n--- POST COLUMN REDUCTION ---\n")
cat("original regressors =", ncol(int.cov), "\n")
cat("after dropping no-variance cols =", sum(post_red$keep_var), "\n")
cat("final post rank =", ncol(x_post), "\n")
cat("-----------------------------\n")

reg.coeff.post <- stats::coef(
    fastglm::fastglm(
        x = x_post,
        y = y[post_filter],
        weights = i.weights[post_filter],
        family = gaussian(link = "identity")
    )
)

    if (anyNA(reg.coeff.post)) {
        stop("Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it.")
    }
    out.y.post <- as.vector(tcrossprod(reg.coeff.post, int.cov.post))
    w.treat.pre <- i.weights * D * (1 - post)
    w.treat.post <- i.weights * D * post
    w.cont <- i.weights * D
    reg.att.treat.pre <- w.treat.pre * y
    reg.att.treat.post <- w.treat.post * y
    reg.att.cont <- w.cont * (out.y.post - out.y.pre)
    eta.treat.pre <- mean(reg.att.treat.pre)/mean(w.treat.pre)
    eta.treat.post <- mean(reg.att.treat.post)/mean(w.treat.post)
    eta.cont <- mean(reg.att.cont)/mean(w.cont)
    reg.att <- (eta.treat.post - eta.treat.pre) - eta.cont
    weights.ols.pre <- i.weights * (1 - D) * (1 - post)
    wols.x.pre <- weights.ols.pre * int.cov.pre
    wols.eX.pre <- weights.ols.pre * (y - out.y.pre) * int.cov.pre
    XpX_pre <- crossprod(wols.x.pre, int.cov.pre)/n
    # Edit
    cat("\n===== XpX_pre DEBUG =====\n")
    cat("nonzero pre OLS weights =", sum(weights.ols.pre != 0), "\n")
    cat("sum(weights.ols.pre) =", sum(weights.ols.pre), "\n")
    cat("dim(wols.x.pre) =", paste(dim(wols.x.pre), collapse = " x "), "\n")
    cat("dim(int.cov) =", paste(dim(int.cov), collapse = " x "), "\n")
    cat("control-pre observations actually entering XpX_pre =", sum(pre_filter), "\n")
    cat("number of original regressors =", ncol(int.cov), "\n")
cat("number of pre regressors used =", ncol(int.cov.pre), "\n")
    cat("rcond not yet checked\n")
    cat("================================\n")
    cat("dim =", dim(XpX_pre), "\n")
    # Edit
    if (base::rcond(XpX_pre) < .Machine$double.eps) {
        stop("The regression design matrix for pre-treatment is singular. Consider removing some covariates.")
    }
    XpX.inv.pre <- solve(XpX_pre)
    asy.lin.rep.ols.pre <- wols.eX.pre %*% XpX.inv.pre
    weights.ols.post <- i.weights * (1 - D) * post
    wols.x.post <- weights.ols.post * int.cov.post
wols.eX.post <- weights.ols.post * (y - out.y.post) * int.cov.post
XpX_post <- crossprod(wols.x.post, int.cov.post)/n
    if (base::rcond(XpX_post) < .Machine$double.eps) {
        stop("The regression design matrix for post-treatment is singular. Consider removing some covariates.")
    }
    XpX.inv.post <- solve(XpX_post)
    asy.lin.rep.ols.post <- wols.eX.post %*% XpX.inv.post
    inf.treat.pre <- (reg.att.treat.pre - w.treat.pre * eta.treat.pre)/mean(w.treat.pre)
    inf.treat.post <- (reg.att.treat.post - w.treat.post * eta.treat.post)/mean(w.treat.post)
    inf.treat <- inf.treat.post - inf.treat.pre
    inf.cont.1 <- (reg.att.cont - w.cont * eta.cont)
    M1_pre <- base::colMeans(w.cont * int.cov.pre)
M1_post <- base::colMeans(w.cont * int.cov.post)

inf.cont.2.post <- asy.lin.rep.ols.post %*% M1_post
inf.cont.2.pre <- asy.lin.rep.ols.pre %*% M1_pre
    inf.control <- (inf.cont.1 + inf.cont.2.post - inf.cont.2.pre)/mean(w.cont)
    reg.att.inf.func <- (inf.treat - inf.control)
    if (boot == FALSE) {
        se.reg.att <- stats::sd(reg.att.inf.func) * sqrt(n -
            1)/(n)
        uci <- reg.att + 1.96 * se.reg.att
        lci <- reg.att - 1.96 * se.reg.att
        reg.boot <- NULL
    }
    if (boot == TRUE) {
        if (is.null(nboot) == TRUE)
            nboot = 999
        if (boot.type == "multiplier") {
            reg.boot <- mboot.did(reg.att.inf.func, nboot)
            se.reg.att <- stats::IQR(reg.boot)/(stats::qnorm(0.75) -
                stats::qnorm(0.25))
            cv <- stats::quantile(abs(reg.boot/se.reg.att), probs = 0.95)
            uci <- reg.att + cv * se.reg.att
            lci <- reg.att - cv * se.reg.att
        }
        else {
            reg.boot <- unlist(lapply(1:nboot, wboot_reg_rc,
                n = n, y = y, post = post, D = D, int.cov = int.cov,
                i.weights = i.weights))
            se.reg.att <- stats::IQR((reg.boot - reg.att))/(stats::qnorm(0.75) -
                stats::qnorm(0.25))
            cv <- stats::quantile(abs((reg.boot - reg.att)/se.reg.att),
                probs = 0.95)
            uci <- reg.att + cv * se.reg.att
            lci <- reg.att - cv * se.reg.att
        }
    }
    if (inffunc == FALSE)
        reg.att.inf.func <- NULL
    call.param <- match.call()
    argu <- mget(names(formals()), sys.frame(sys.nframe()))
    boot.type <- ifelse(argu$boot.type == "multiplier", "multiplier",
        "weighted")
    boot <- ifelse(argu$boot == TRUE, TRUE, FALSE)
    argu <- list(panel = FALSE, boot = boot, boot.type = boot.type,
        nboot = nboot, type = "or")
    ret <- (list(ATT = reg.att, se = se.reg.att, uci = uci, lci = lci,
        boots = reg.boot, att.inf.func = reg.att.inf.func, call.param = call.param,
        argu = argu))
    class(ret) <- "drdid"
    return(ret)
}