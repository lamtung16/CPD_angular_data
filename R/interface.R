#' Run change point detection
#'
#' @param signal A numeric matrix
#' @param pen A numeric penalty value
#' @param n_states Number of states
#' @return An integer vector of change points
#' @export
apart_rcpp <- function(signal, pen, n_states) {
  .Call('_apartruptures_apart_rcpp', PACKAGE = 'apartruptures', signal, pen, n_states)
}
