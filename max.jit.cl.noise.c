/* 
	Copyright 2001-2005 - Cycling '74
	Joshua Kit Clayton jkc@cycling74.com	
*/

#include "jit.common.h"
#include "max.jit.mop.h"

typedef struct _max_jit_cl_noise 
{
	t_object		ob;
	void			*obex;
} t_max_jit_cl_noise;

t_jit_err jit_cl_noise_init(void); 

void *max_jit_cl_noise_new(t_symbol *s, long argc, t_atom *argv);
void max_jit_cl_noise_free(t_max_jit_cl_noise *x);
void max_jit_cl_noise_outputmatrix(t_max_jit_cl_noise *x);
void *max_jit_cl_noise_class;
		 	
void main(void)
{	
	void *p,*q;
	
	jit_cl_noise_init();	
	setup(&max_jit_cl_noise_class, max_jit_cl_noise_new, (method)max_jit_cl_noise_free, (short)sizeof(t_max_jit_cl_noise), 
		0L, A_GIMME, 0);

	p = max_jit_classex_setup(calcoffset(t_max_jit_cl_noise,obex));
	q = jit_class_findbyname(gensym("jit_cl_noise"));    
    max_jit_classex_mop_wrap(p,q,MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX|MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX); 		
    max_jit_classex_standard_wrap(p,q,0);
	max_addmethod_usurp_low((method)max_jit_cl_noise_outputmatrix, "outputmatrix");	
    addmess((method)max_jit_mop_assist, "assist", A_CANT,0);
}

void max_jit_cl_noise_outputmatrix(t_max_jit_cl_noise *x)
{

	long outputmode=max_jit_mop_getoutputmode(x);
	void *mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop);
	t_jit_err err;	
	
	if (outputmode&&mop) { //always output unless output mode is none
		if (outputmode==1) {
			if (err=(t_jit_err)jit_object_method(
												 max_jit_obex_jitob_get(x), 
												 _jit_sym_matrix_calc,
												 jit_object_method(mop,_jit_sym_getinputlist),
												 jit_object_method(mop,_jit_sym_getoutputlist)))						
			{
				jit_error_code(x,err); 
			} else {
				max_jit_mop_outputmatrix(x);
			}
		} else {
			max_jit_mop_outputmatrix(x);
		}
	}	
}

void max_jit_cl_noise_free(t_max_jit_cl_noise *x)
{
	max_jit_mop_free(x);
	jit_object_free(max_jit_obex_jitob_get(x));
	max_jit_obex_free(x);
}

void *max_jit_cl_noise_new(t_symbol *s, long argc, t_atom *argv)
{
	t_max_jit_cl_noise *x;
	void *o;

	if (x=(t_max_jit_cl_noise *)max_jit_obex_new(max_jit_cl_noise_class,gensym("jit_cl_noise"))) {
		if (o=jit_object_new(gensym("jit_cl_noise"))) {
			max_jit_mop_setup_simple(x,o,argc,argv);			
			max_jit_attr_args(x,argc,argv);
		} else {
			jit_object_error((t_object *)x,"jit.cl.noise: could not allocate object");
			freeobject(x);
			x = NULL;
		}
	}
	return (x);
}
