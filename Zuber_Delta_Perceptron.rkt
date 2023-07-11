#lang racket

(define learning-rate 0.01)
(define num-epochs 1000000)

;; Delta rule perceptron function
(define (delta-rule-perceptron weights inputs target)
  (let* ((output (dot-product weights inputs))
         (error (- target output))
         (delta (* learning-rate error))
         (updated-weights (map + weights (map (lambda (input) (* delta input)) inputs))))
    updated-weights))

;; Dot product function
(define (dot-product vec1 vec2)
  (apply + (map * vec1 vec2)))

;; Training function
(define (train perceptron inputs targets)
  (let loop ((weights perceptron) (epochs 0))
    (if (>= epochs num-epochs)
        weights
        (let* ((index (random (length inputs)))
               (input (list-ref inputs index))
               (target (list-ref targets index))
               (updated-weights (delta-rule-perceptron weights input target)))
          (loop updated-weights (add1 epochs))))))

;; Classification function
(define (classify perceptron inputs)
  (map (lambda (input) (if (> (dot-product perceptron input) 0) 'A 'B)) inputs))

;; Creating the datasets 
(define data-A '((2 4) (3 5) (4 6) (5 7) (6 8) (7 9) (8 10) (9 11) (10 12) (11 13)))
(define data-B '((1 1) (2 2) (3 3) (4 4) (5 5) (6 6) (7 7) (8 8) (9 9) (10 10)))

;; Splitting the data into training and test sets
(define training-A (take data-A (quotient (length data-A) 2)))
(define test-A (drop data-A (quotient (length data-A) 2)))
(define training-B (take data-B (quotient (length data-B) 2)))
(define test-B (drop data-B (quotient (length data-B) 2)))

;; Combining training and test data
(define training-data (append training-A training-B))
(define test-data (append test-A test-B))

;; Creating target labels
(define target-A (make-list (length training-A) 1))
(define target-B (make-list (length training-B) -1))
(define target-data (append target-A target-B))

;; Initializing perceptron weights
(define initial-weights (make-list (length (car training-data)) 0))

;; Training the perceptron
(define trained-weights (train initial-weights training-data target-data))

;; Classifying test data
(define classification-results (classify trained-weights test-data))

;; Printing the classification results
(displayln "Test Results:")
(for-each (lambda (data result)
            (displayln (format "Data: ~a, Result: ~a" data result)))
          test-data
          classification-results)
