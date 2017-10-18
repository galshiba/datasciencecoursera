## The programs (part of the Coursera course on R programming
## will focus on caching an inverted matrix.

## makeCacheMatrix will generate an inverted matrix and cache the
## result

makeCacheMatrix <- function(x = matrix()) {
	m <- NULL
	set <- function(y) {
		x <<- y
		m <<- NULL
	}
	get <- function() x
	setinv <- function(inv) m <<- inv
	getinv <- function() m
	list(set = set, get = get,
		setinv = setinv,
		getinv = getinv)
}


## cacheSolve computes the inverse of the special "matrix" returned by
## makeCacheMatrix. If the inverse has already been computed, then the 
## program will retrieve the inverse from the cache.

cacheSolve <- function(x, ...) {
	## Return a matrix that is the inverse of 'x'
	m <- x$getinv()
	if(!is.null(m)) {
		message("getting cached data")
		return(m)
	}
	data <- x$get()
	m <- solve(data,...)
	x$setinv(m)
	m
}
