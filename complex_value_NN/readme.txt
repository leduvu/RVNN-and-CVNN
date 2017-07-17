Forward Processing and Learning Dynamics

CVNN (complex value neural network)

T  = transpose
w  = weight			(matrix) is given
z  = signal [x1,x2,...]		(vector) has to be T
z^ = teacher signal 		(vector) is given for I and O, has to be T
H  = hidden neurons 	= 25
I  = Input neurons 	= 16
O  = Output neurons	= 16
k  = learning constant 	= 0.01
x  = Element of signal vector
x^os = Element of signal output teacher vector
S  = number of signals


Propagation function is used to transport values through the neurons of a 
neural net's layers. Usually, the input values are added up and passed to an 
activation function, which generates an output.
The resulting value is compared with a certain threshold value by the neuron's 
activation function. If the input exceeds the threshold value, the neuron will 
be activated, otherwise it will be inhibited.
If activated, the neuron sends an output on its outgoing weights to all connected neurons and so on. 

A) FORWARD SIGNAL PROCESSING
activation function - to connect weights with signals: f(x) = tanh(x)h(|z|)exp(j arg z)
zHidden = f( w11*z1 + w12*z2 +...)	wHid zIn  (vector again)
zOut 	= f(			 )	wOut zHid (vector again)

additional weitghs (tresholds): 	wh I+1		wR ho H+2
additional formal constants: 		zI+1 = 1 	zH+1 = 1


B) ERROR BACKPROPAGATION LEARNING		
error function = E
	for a set of input and output teacher signals (zI,zO) to obtain 
	the learning dynamics
	(aim is a most possible minimal value)

1. get some input
2. output will be compared with the desired output
3. the error will be backpropageded from the output to the input
4. -> the weights will be changed

E 	= error
S	= number of signals
2O	= doubled number of output neurons = 32
xo(Z^I)	= output    wiki != paper (because of abs -> can be switched)
X^os	= target    wiki != paper
1/2	= just for simpplifying the derivation

Least mean square function is used to calculate the error value.
There is a set of teacher input signal vectors zI (counter s) and each of these 
signal vectors have elements x (counter o).
	
update function via partial derivation



Ressources:
Generalization Characteristics of Complex-Valued Feedforward 
	Neural Netwarks in Relation to Signal Coherence (2012) Akira Hirose, Shotaro Yoshida
https://de.wikipedia.org/wiki/Backpropagation
https://de.wikipedia.org/wiki/Differentialrechnung
http://www.nnwj.de/propagation-function-term.html