# A-Threshold-Based-Editing-Approach-for-Multi-label-Instance-Selection
A common strategy for improving the efficiency of instance-based classifiers,
while preserving high accuracy, is to reduce the size of the training set by
replacing it with a smaller yet representative subset. This is typically accom-
plished through data reduction techniques, which have proven highly effective
in single-label classification tasks. Their applicability, however, is far less
straightforward in multi-label scenarios, where each instance may be associated
with multiple classes simultaneously. In this paper, we extend a widely used
single-label data reduction method—the Edited Nearest Neighbor rule to the
multi-label domain. In single-label classification, Editted Nearest Neigbour rile
aims to eliminate noisy and borderline instances, thereby producing a cleaner
dataset with more clearly defined decision boundaries. The underlying assump-
tion is that instances whose class differs from that of most of their neighbors are
likely to be noise and should be removed. We develop a new type of multi-label
data processing and classification method to help improve prediction perfor-
mance by eliminating the effect of noisy or ambiguous training examples on our
classification method. The proposed method combines a new Algorithm named
Threshold-Based Multi-Label Editing Algorithm filtering with a Binary Rele-
vance k-Nearest Neighbors Classifier and evaluates the performance using the
Hamming Loss metric. In multi-label data, the boundaries of multi-label cate-
gories are inherently less clear, making it more difficult to identify noise in the
data. Nonetheless, we suggest that examples with labelsets that differ signifi-
cantly from those found within the local neighbourhood are indicative of noise.
Therefore, removing these noisy instances will allow for a condensed training set
while preserving some of the underlying structural properties of the data. The
new algorithm named Threshold-Based Multi-Label Editing Algorithm (TME)
was developed based on the above concept. The performance of algorithm was
assessed through experimental tests on nine (9) diverse multi-label datasets. Our
findings show that the proposed algorithm significantly reduce the size of the
datasets, while maintaining classification accuracy.
