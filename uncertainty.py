# Copyright (c) 2023 Chris H. Zhao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""

This module provides 'uncertainty' class that helps address error calculations in Basic Physics Lab and Physical Chemistry Lab.

Requirements
============

torch, sympy

"""
import torch
import math
import sympy
import functools
for i,j in zip(['sqrt','log','exp','sin','cos','tan','sinh','cosh','tanh','arcsin','arccos','arctan','arcsinh','arccosh','arctanh','asin','acos','atan','asinh','acosh','atanh'],['sqrt','log','exp','sin','cos','tan','sinh','cosh','tanh','asin','acos','atan','asinh','acosh','atanh','asin','acos','atan','asinh','acosh','atanh']):
	setattr(sympy.core.expr.Expr,i,lambda self:getattr(sympy,j)(self))
sympy.core.expr.Expr.log10=lambda self:sympy.log(self)/sympy.log(10)
sympy.core.expr.Expr.qbrt=lambda self:self**sympy.Rational(1,3)
class uncertainty():
	"""

	'uncertainty' is a class for uncertainty calculation needed in Basic Physics Lab and Physical Chemistry Lab.
	It calculates error on the fly and generates latex expression on request.
	This calculation has taken the correlation induced by repeated presence of the same variable into account.
	It is compatible with vector operation classes like numpy.ndarray, pandas.DataFrame,... so
	batch calculation can be easily performed on tabulated experimental data.
	The most useful functions/methods implemented are: uncertainty.latex, uncertainty.mean, uncertainty.lsmr, uncertainty.__str__

	Usage
	=====

	uncertainty(value:float=0, error:float=0, name:str="") -> uncertainty_object
	uncertainty(value, error, name) returns uncertainty object "name = value ± error"
	uncertainty(uncertainty_object) returns the object itself
	uncertainty(uncertainty_object, name=name) returns renamed uncertainty object (uncertainty_object.rename(name) doesn't return)

	Examples
	========

	>>>	from uncertainty import uncertainty
	>>>	import numpy as np

	>>> a=uncertainty(1, 0.1,'a') # generates an object representing 1.0 ± 0.1
	>>> a
	uncertainty(1.0,0.1,'a')
	>>> print(a)
	a = 1.0 ± 0.1
	>>> print(a-a) # correlation considered
	0.0000 ± 0.0000
	>>> print(a-uncertainty(1, 0.1)) # no correlation
	0.0 ± 0.2

	>>> print(np.sin(a)) # advanced operations are supported
	0.84 ± 0.06

	>>> print(np.sum(np.sin(np.array([a, -a])))) # array operations are supported with correlation considered
	0.0000 ± 0.0000

	>>> print(np.sum(np.array([uncertainty(i, 0.1) for i in range(100)]))) # batch operation on array
	4950 ± 2

	See Also
	========

    uncertainty.rename, EF.EF

	"""
	def __new__(cls,value=0,error=0,*_,**__):
		return value if isinstance(value,uncertainty) else object.__new__(uncertainty)
	def __init__(self,value=0,error=0,name=""):
		if name:
			self.name=name
		if isinstance(value,uncertainty):
			return
		elif isinstance(value,torch.Tensor) and isinstance(error,sympy.core.expr.Expr):
			setattr(self,'0',value)
			setattr(self,'1',error)
			return
		try:
			iter(value)
			iter(error)
		except:
			self.value=value
			self.error=error
			self.name=name
			setattr(self,'0',torch.tensor(value,dtype=torch.float64,requires_grad=True))
			getattr(self,'0').associate=self
			setattr(self,'1',sympy.symbols("$%s$"%name))
			return
		raise AssertionError("0d input required!")
	def iter_leaf(self):
		"""

		'uncertainty.iter_leaf' returns an iterator over all leaf objects of the object's computational graph.

		Usage
		=====

		uncertainty.iter_leaf(uncertainty object) -> generator
		uncertainty_object.iter_leaf() -> generator

		"""
		if getattr(self,'0').grad_fn is None:
			yield self
			return
		stack=[getattr(self,'0').grad_fn]
		visited=[]
		while stack:
			i=stack.pop()
			if i.__class__.__name__=="AccumulateGrad" and i not in visited:
				visited.append(i)
				yield i.variable.associate
			for j in i.next_functions:
				if j[0] is not None:
					stack.append(j[0])
	def eval(self):
		"""

		'uncertainty.eval' calculates the object's error and fill in the object's name with empty string.
		Normally this function is not needed as it is automatically called during output.

		Usage
		=====

		uncertainty.eval(uncertainty_object) -> None
		uncertainty_object.eval() -> None

		"""
		try:
			self.name
		except:
			self.name=""
		self.value=getattr(self,'0').item()
		getattr(self,'0').backward(retain_graph=True)
		self.error=math.sqrt(sum([[(getattr(i,'0').grad*i.error)**2,getattr(i,'0').grad.zero_()][0] for i in uncertainty.iter_leaf(self)]))
	def rename(self,name=""):
		"""

		'uncertainty.rename' changes object's name without returning the object.
		Use uncertainty(uncertainty_object, name=name) if returning behavior is desired.

		Usage
		=====

		uncertainty.rename(uncertainty_object, name="") -> None
		uncertainty_object.eval(name="") -> None

		See Also
		========

		uncertainty

		"""
		self.name=name
	def latex(self):
		"""

		'uncertainty.latex' automatically generates latex formula that can be used inside equation, displaymath, ...
		This is the highlight of this class that greatly helps reduce workload.

		Usage
		=====

		uncertainty.latex(uncertainty_object) -> str
		uncertainty_object.latex() -> str

		Example
		=======

		>>>	from uncertainty import uncertainty

		>>> a, b=uncertainty(1, 0.1, 'a'), uncertainty(2, 0.01, 'b')
		>>> print(uncertainty(a*b, name='c').latex()) # simple calculation is supported
		\\sigma_{c} = \\sqrt{\\sigma_{a}^{2} \\cdot \\left(\\frac{\\mathrm{d}c}{\\mathrm{d}a}\\right)^{2} + \\sigma_{b}^{2} \\cdot \\left(\\frac{\\mathrm{d}c}{\\mathrm{d}b}\\right)^{2}} = \\sqrt{\\sigma_{a}^{2} \\cdot b^{2} + \\sigma_{b}^{2} \\cdot a^{2}} = \\sqrt{1.0^{2} \\cdot 0.01^{2} + 2.00^{2} \\cdot 0.1^{2}} = 0.2002498439450079 \\approx 0.3

		>>> print(uncertainty.mean([1,2,3], 0.1,name='x').latex()) # calculation that generates additional error is supported
		\\sigma_{\\bar{x}} = \\sqrt{\\sigma_{x 0}^{2} \\cdot \\left(\\frac{\\bar{\\mathrm{d}x}}{\\mathrm{d}x_{0}}\\right)^{2} + \\sigma_{x 1}^{2} \\cdot \\left(\\frac{\\bar{\\mathrm{d}x}}{\\mathrm{d}x_{1}}\\right)^{2} + \\sigma_{x 2}^{2} \\cdot \\left(\\frac{\\bar{\\mathrm{d}x}}{\\mathrm{d}x_{2}}\\right)^{2} + \\sigma_{\\bar{x}}^{2} \\cdot 1^{2}} = \\sqrt{\\frac{\\sigma_{x 0}^{2}}{9} + \\frac{\\sigma_{x 1}^{2}}{9} + \\frac{\\sigma_{x 2}^{2}}{9} + \\sigma_{\\bar{x}}^{2}} = \\sqrt{\\frac{0.1^{2}}{9} + \\frac{0.1^{2}}{9} + \\frac{0.1^{2}}{9} + 0.6^{2}} = 0.5802298395176404 \\approx 0.6

		See Also
		========

		uncertainty.__str__, uncertainty.__repr__

		"""
		self.eval()
		for i in self.iter_leaf():
			i.eval()
		dx=sympy.symbols("$d$"+self.name)
		return functools.reduce(lambda x,y:y(x),[" = ".join([sympy.latex(sympy.symbols("sigma_"+self.name)),
			sympy.latex(sympy.sqrt(sum(sympy.UnevaluatedExpr(dx/sympy.symbols("$d$"+i.name))**2*sympy.symbols("sigma_%s"%i.name)**2 for i in self.iter_leaf())),parenthesize_super=False,ln_notation=True,mul_symbol='dot'),
			sympy.latex(sympy.sqrt(sum(sympy.diff(getattr(self,'1'),getattr(i,'1')).simplify()**2*sympy.symbols("sigma_%s"%i.name)**2 for i in self.iter_leaf())).subs({getattr(i,'1'):sympy.UnevaluatedExpr(sympy.symbols(i.name)) for i in self.iter_leaf()}),parenthesize_super=False,ln_notation=True,mul_symbol='dot'),
			functools.reduce(lambda x,y:y(x),[sympy.latex(sympy.sqrt(sum(sympy.diff(getattr(self,'1'),getattr(i,'1')).simplify()**2*sympy.symbols("$sigma_%s$"%i.name)**2 for i in self.iter_leaf())),parenthesize_super=False,ln_notation=True,mul_symbol='dot')]+[lambda x,i=i:x.replace(sympy.latex(sympy.symbols("$%s$"%i.name),parenthesize_super=False,ln_notation=True,mul_symbol='dot'),str(i).split()[-3].replace('(','')+'×'+str(i).split('×')[-1] if '×' in str(i) else str(i).split()[-3]) for i in self.iter_leaf()]+[lambda x,i=i:x.replace(sympy.latex(sympy.symbols("$sigma_%s$"%i.name),parenthesize_super=False,ln_notation=True,mul_symbol='dot'),str(i).split()[-1].replace(')','')) for i in self.iter_leaf()]),
			str(self.error)])+" \\approx "+str(self).split()[-1].replace(')',''),lambda x:x.replace("$d$","\\mathrm{d}"),lambda x:x.replace("$slope$","\\mathrm{slope}"),lambda x:x.replace("$intercept$","\\mathrm{intercept}")])
	def __str__(self):
		"""

		'uncertainty.__str__' generates rounded expression that is conveniently copied into the lab report.
		Use of scientific notation is automatically decided based on the magnitude of input and its error.

		Usage
		=====

		uncertainty.__str__(uncertainty_object) -> str
		uncertainty_object.__str__() -> str
		str(uncertainty_object) -> str
		print(uncertainty_object) -> None

		Example
		=======

		>>>	from uncertainty import uncertainty

		>>> print(uncertainty(1, 0.1))
		1.0 ± 0.1
		>>> print(uncertainty(1, 0.1, 'a'))
		a = 1.0 ± 0.1
		>>> print(uncertainty(0.00001, 0.000001))
		(10 ± 1)×10^-6
		>>> print(uncertainty(-0.00001, 0.1))
		-0.0 ± 0.1
		>>> print(uncertainty(-1000, 10))
		-(100 ± 1)×10^1

		See Also
		========

		uncertainty.latex, uncertainty.__repr__

		"""
		self.eval()
		value,error=float(self.value),float(self.error)
		if self.name:
			name=self.name+" = "
		else:
			name=""
		if error==0:
			if value==0:
				return name+"0.0000 ± 0.0000"
			dnum=-math.floor(math.log(abs(value),10))
			return name+"%%.%df ± %%.%df"%(dnum+4,dnum+4)%(value,0) if 0<=dnum<=4 else name+"-"*(value<0)+"(%.4f ± 0.0000)×10^%d"%(abs(value)*10**dnum,-dnum)
		else:
			enum=-math.floor(math.log(error,10))
			return name+"%%.%df ± %%.%df"%(enum,enum)%(value,math.ceil(error*10**enum)/10**enum) if 0<=enum<=4 else name+"-"*(value<0)+"(%d ± %d)×10^%d"%(abs(value)*10**enum,math.ceil(error*10**enum),-enum)
	def __repr__(self):
		"""

		'uncertainty.__repr__' generates python representation of an uncertainty object.
		This is opposed to __str__ that generates user-friendly expression.

		Usage
		=====

		uncertainty.__repr__(uncertainty_object) -> str
		uncertainty_object.__repr__() -> str
		repr(uncertainty_object) -> str

		Example
		=======

		>>>	from uncertainty import uncertainty

		>>> uncertainty(1, 0.1)
		uncertainty(1.0, 0.1)
		>>> uncertainty(1, 0.1, 'a')
		uncertainty(1.0, 0.1, 'a')

		See Also
		========

		uncertainty.latex, uncertainty.__str__

		"""
		self.eval()
		if self.name:
			return "uncertainty(%s, %s, %s)"%(repr(self.value),repr(self.error),repr(self.name))
		else:
			return "uncertainty(%s, %s)"%(repr(self.value),repr(self.error))
	@staticmethod
	def mean(x,sx=0,name=""):
		"""

		'uncertainty.mean' calculates average of multiple figures and considers extra error
		introduced by averaging multiple measurements (bias-variation decomposition).
		It also automatically name the input array (if the objects named are not leaf nodes, the naming will not affect latex generation).
		Such error can be accounted for by fitting in MATLAB with '0*x+a' and see the confidence interval of a.

		Usage
		=====

		uncertainty.mean(x:iterable, sx:float=0, name:str="") -> uncertainty_object
		uncertainty.mean(x, sx, name) returns uncertainty object \\bar{name} that is average of list of floats x and uniform error sx
		uncertainty.mean(x, name=name) returns uncertainty object \\bar{name} that is average of list of uncertainty_objects x
		that doesn't need to have uniform error (weighting based on error 1/e_i^2 is not implemented yet)

		Example
		=======

		>>>	from uncertainty import uncertainty
		>>>	import numpy as np

		>>> print(uncertainty.mean(np.arange(10), 0.1, name='x')) # uniform error
		xbar = 4.5 ± 1.0

		>>> print(uncertainty.mean(np.array([uncertainty(i, i*0.1) for i in np.arange(10)]), name='x')) # not uniform error
		xbar = 4.5 ± 1.0

		See Also
		========

		uncertainty.lsmr

		"""
		if any(isinstance(i,uncertainty) for i in x):
			x=[uncertainty(i,name=name+'_'+str(j)) for j,i in enumerate(x)] if name else list(map(uncertainty,x))
		else:
			x=[uncertainty(i,sx,name+'_'+str(j)) for j,i in enumerate(x)]
		a=uncertainty(sum(x)/len(x),name=name) if name else sum(x)/len(x)
		sx1=math.sqrt(sum((float(i)-float(a))**2 for i in x)/(len(x)-1)/len(x))
		return uncertainty(a+uncertainty(0,sx1,'%sbar'%name),name='%sbar'%name)
	@staticmethod
	def lsmr(x,y,sx=0,sy=0,namex="",namey=""):
		"""

		'uncertainty.lsmr' calculates 'least-square mean regression' and considers extra error
		introduced by linear regression (what one have learnt in ACGDC).
		It also automatically name the input arrays (if the objects named are not leaf nodes, the naming will not affect latex generation).
		Such error can be accounted for by fitting in MATLAB with 'a*x+b' and see the confidence interval of a,b.

		Usage
		=====

		uncertainty.lsmr(x:iterable, y:iterable, sx:float=0, sy:float=0, namex:str="", namey:str="") -> Tuple[uncertainty_object, uncertainty_object, float]
		uncertainty.lsmr(x, y, sx, sy, name) returns uncertainty objects slope_{name}, intercept_{name} and R^2 of lists of floats x, y and uniform error sx, sy
		uncertainty.lsmr(x, name=name) returns uncertainty objects slope_{name}, intercept_{name} and R^2 of list of uncertainty_objects x, y
		that doesn't need to have uniform error (weighting based on error 1/e_i^2 is not implemented yet)

		Example
		=======

		>>>	from uncertainty import uncertainty
		>>>	import numpy as np

		>>> print(uncertainty.lsmr(np.arange(10), np.arange(10), sy=0.1, namey='y')) # uniform error for both x, y
		(uncertainty(1.0, 0.011009637651263606, '$slope$_y'), uncertainty(0.0,0.05877538136452587, '$intercept$_y'), 1.0)

		>>> print(uncertainty.lsmr([uncertainty(i**2, i*0.1) for i in range(10)], np.arange(10), sy=0.1, namey='y')) # uniform error for y but not x
		(uncertainty(0.10297482837528604, 0.010275886130810444, '$slope$_y'), uncertainty(1.565217391304348, 0.4012246982843369, '$intercept$_y'), 0.9267734553775743)

		>>> print(uncertainty.lsmr([uncertainty(i**2, i*0.1) for i in range(10)], [uncertainty(i,i*0.1) for i in range(10)], namey='y')) # not uniform error for both x, y
		(uncertainty(0.10297482837528604, 0.01304621475013945, '$slope$_y'), uncertainty(1.565217391304348, 0.4300705016777366, '$intercept$_y'), 0.9267734553775743)

		See Also
		========

		uncertainty.lsmr

		"""
		assert (n:=len(x))==len(y),"How am I supposed to do a fit with len(x)!=len(y)?!"
		if any(isinstance(i,uncertainty) for i in x) or any(isinstance(i,uncertainty) for i in y):
			x=[uncertainty(i,name=namex+'_'+str(j)) for j,i in enumerate(x)] if namex else list(map(uncertainty,x))
			y=[uncertainty(i,name=namey+'_'+str(j)) for j,i in enumerate(y)] if namey else list(map(uncertainty,y))
		else:
			x=[uncertainty(i,sx,namex+'_'+str(j)) for j,i in enumerate(x)]
			y=[uncertainty(i,sy,namey+'_'+str(j)) for j,i in enumerate(y)]
		a=(n*sum(i*j for i,j in zip(x,y))-sum(x)*sum(y))/(n*sum(i**2 for i in x)-sum(x)**2)
		b=(sum(i**2 for i in x)*sum(y)-sum(x)*sum(i*j for i,j in zip(x,y)))/(n*sum(i**2 for i in x)-sum(x)**2)
		rsquare=(n*sum(float(i)*float(j) for i,j in zip(x,y))-sum(map(float,x))*sum(map(float,y)))**2/(n*sum(float(i)**2 for i in x)-sum(map(float,x))**2)/(n*sum(float(i)**2 for i in y)-sum(map(float,y))**2)
		sy1=math.sqrt(sum((float(a)*i+float(b)-j)**2 for i,j in zip(map(float,x),map(float,y)))/(n-2))
		return uncertainty(a+uncertainty(0,(lambda x:math.sqrt(n*sy1**2/(n*sum(i**2 for i in x)-sum(x)**2)))([i.value for i in x]),'$slope$_%s'%namey),name='$slope$_%s'%namey),uncertainty(b+uncertainty(0,(lambda x:math.sqrt(sum(i**2 for i in x)*sy1**2/(n*sum(i**2 for i in x)-sum(x)**2)))([i.value for i in x]),'$intercept$_%s'%namey),name='$intercept$_%s'%namey),rsquare
for op in ['__add__','__sub__','__mul__','__truediv__','__pow__','__radd__','__rsub__','__rmul__','__rtruediv__','__rpow__','__pos__','__neg__','__abs__','sqrt','qbrt','log','log10','exp','sin','cos','tan','sinh','cosh','tanh','arcsin','arccos','arctan','arcsinh','arccosh','arctanh','asin','acos','atan','asinh','acosh','atanh']:
	def operator(self,*_,op=op):
		"""

		'uncertainty.`operator`' performs calculation on uncertainty objects.
		These operator functions are automatically generated.
		Their semantics are the same as normal python operations, in line with one's program sense.

		"""
		try:
			return uncertainty(getattr(getattr(self,'0'),op)(*(getattr(i,'0') if isinstance(i,uncertainty) else i for i in _)),getattr(getattr(self,'1'),op)(*(getattr(i,'1') if isinstance(i,uncertainty) else i for i in _)))
		except:
			return NotImplemented
	setattr(uncertainty,op,operator)
for op in ['__lt__','__le__','__eq__','__ne__','__gt__','__ge__','__int__','__float__']:
	setattr(uncertainty,op,lambda self,*_,op=op:getattr(getattr(self,'0'),op)(*(getattr(i,'0') for i in _)))