/* 
	Copyright 2001 - Cycling '74
	Joshua Kit Clayton jkc@cycling74.com	
*/
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

/// opencl stuff
///
///
///

////////////////////////////////////////////////////////////////////////////////////////////////////


#define DEBUG_INFO         (0)             // enable debug info
#define SEPARATOR          ("----------------------------------------------------------------------\n")

#define COMPUTE_KERNEL_FILENAME                 ("jit.cl.noise_kernel.cl")
#define COMPUTE_KERNEL_COUNT                    4

////////////////////////////////////////////////////////////////////////////////////////////////////


static const char * ComputeKernelMethods[COMPUTE_KERNEL_COUNT] =
{
	"GradientNoiseArray2d",
	"MonoFractalArray2d",
	"TurbulenceArray2d",
	"RidgedMultiFractalArray2d",
};



#include "jit.common.h"

typedef struct _jit_cl_noise 
{
	t_object				ob;
	long					mode;
	long					post;
	
	// openCL stuff:
	cl_context                               ComputeContext;
	cl_command_queue                         ComputeCommands;
	cl_kernel                                ComputeKernel;
	cl_program                               ComputeProgram;
	cl_device_id                             ComputeDeviceId;
	cl_device_type                           ComputeDeviceType;
	cl_mem                                   ComputeResult;
	
	char* HostImageBuffer;             
	 int Width                                ;
	 int Height                               ;
	
	 float scale                              ;
	 float bias[2]                            ;
	 float biasX;
	 float biasY;
	 float lacunarity                         ;
	 float increment                          ;
	 float octaves                            ;
	 float amplitude                          ;
	
	
	 //int ActiveKernel                   ;
	
	 size_t TextureTypeSize       ;

	 cl_kernel ComputeKernels[COMPUTE_KERNEL_COUNT];
	 int ComputeKernelWorkGroupSizes[COMPUTE_KERNEL_COUNT];

	////////////////////////////////////////////////////////////////////////////////
	

	
} t_jit_cl_noise;

void *_jit_cl_noise_class;

t_jit_err jit_cl_noise_init(void);
t_jit_cl_noise *jit_cl_noise_new(void);
void jit_cl_noise_free(t_jit_cl_noise *x);
t_jit_err jit_cl_noise_matrix_calc(t_jit_cl_noise *x, void *inputs, void *outputs);

void jit_cl_noise_scale(t_jit_cl_noise *x, t_symbol *s, long argc, t_atom *argv);
void jit_cl_noise_bias(t_jit_cl_noise *x, t_symbol *s, long argc, t_atom *argv);


////////////////////////////////////////////////////////////////////////////////
///////				OpenCL				///////				OpenCL				///////				OpenCL				///////				OpenCL				
///////				OpenCL				///////				OpenCL				///////				OpenCL				///////				OpenCL				
///////				OpenCL				///////				OpenCL				///////				OpenCL				///////				OpenCL				

static int 
FloorPow2(int n)
{
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
}


static int LoadTextFromFile(
							const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;
	
	short path; 
	char name[1024];
	char full_name[1024];
	char conform_name[1024];
	long type;
	
	strcpy(name,file_name);

	if (locatefile_extended(name,&path,&type,&type,-1))
	{
		error("Can't find OpenCL kernel file: %s",name);
		return -1;
	}
	else
		if(path_topathname(path, name, full_name))
		{
			error("Pathname error: %s",full_name);	
			return -1;
		}
			else
				if(path_nameconform(full_name, conform_name,PATH_STYLE_SLASH, PATH_TYPE_BOOT))
				{
					error("conform error: %s",conform_name);	
					return -1;
				}
		


	
    *string_len = 0;
    fd = open(conform_name, O_RDONLY);
    if (fd == -1)
    {
        error("Error opening file %s\n", conform_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret)
    {
        error("Error reading status for file %s\n", conform_name);
        return -1;
    }
    file_len = file_status.st_size;
	
    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = read(fd, *result_string, file_len);
    if (!ret)
    {
        error("Error reading from file %s\n", conform_name);
        return -1;
    }
	
    close(fd);
	
    *string_len = file_len;
    return 0;
}

static void 
CreateBuffers(t_jit_cl_noise *x)
{
    if (x->HostImageBuffer)
        free(x->HostImageBuffer);
	
    x->HostImageBuffer = malloc(x->Width * x->Height * x->TextureTypeSize * 4);
    memset(x->HostImageBuffer, 0, x->Width * x->Height * x->TextureTypeSize * 4);
	
}

static int
Recompute(t_jit_cl_noise *x)
{
    void *values[10];
    size_t sizes[10];
    size_t global[2];
    size_t local[2];
	
    int arg = 0;
    int err = 0;
    float bias[2] = { fabs(x->biasX), fabs(x->biasY) };
    float scale[2] = { fabs(x->scale), fabs(x->scale) };
	
    unsigned int v = 0, s = 0;
    values[v++] = &x->ComputeResult;
    values[v++] = bias;
    values[v++] = scale;
    if(x->mode > 0)
    {
        values[v++] = &x->lacunarity;
        values[v++] = &x->increment;
        values[v++] = &x->octaves;
    }
    values[v++] = &x->amplitude;
	
    sizes[s++] = sizeof(cl_mem);
    sizes[s++] = sizeof(float) * 2;
    sizes[s++] = sizeof(float) * 2;
    if(x->mode > 0)
    {
        sizes[s++] = sizeof(float);
        sizes[s++] = sizeof(float);
        sizes[s++] = sizeof(float);
    }
    sizes[s++] = sizeof(float);
	
    err = CL_SUCCESS;
    for (arg = 0; arg < s; arg++)
    {
        err |= clSetKernelArg(x->ComputeKernels[x->mode], arg, sizes[arg], values[arg]);
    }
	
    if (err)
        return -10;
	
    global[0] = x->Width;
    global[1] = x->Height;
	
    local[0] = x->ComputeKernelWorkGroupSizes[x->mode];
    local[1] = 1;
	
	
    err = clEnqueueNDRangeKernel(x->ComputeCommands, x->ComputeKernels[x->mode], 2, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        error("Failed to enqueue kernel! %d\n", err);
        return err;
    }
	
    err = clEnqueueReadBuffer( x->ComputeCommands, x->ComputeResult, CL_TRUE, 0, 
							  x->Width * x->Height * x->TextureTypeSize * 4, 
							  x->HostImageBuffer, 0, NULL, NULL );      
    if (err)
        return -5;

	
    return CL_SUCCESS;
}

////////////////////////

////////////////////////////////////////////////////////////////////////////////

static int 
CreateComputeResult(t_jit_cl_noise *x)
{
	
    int err = 0;
    //if (x->post!=0) post(SEPARATOR);
	
	
    if (x->post!=0) post("Allocating compute result buffer...\n");
    x->ComputeResult = clCreateBuffer(x->ComputeContext, CL_MEM_WRITE_ONLY, x->TextureTypeSize * 4 * x->Width * x->Height, NULL, &err);
    if (!x->ComputeResult || err != CL_SUCCESS)
    {
        error("Failed to create OpenCL array! %d\n", err);
        return EXIT_FAILURE;
    }
	
    return CL_SUCCESS;
}

static int 
SetupComputeKernels(t_jit_cl_noise *x)
{
    int err = 0;
    char *source = 0;
    size_t length = 0;
	
    if (x->post!=0) post("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME);    
    err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME, &source, &length);
    if (err)
        return -8;
	
    // Create the compute program from the source buffer
    //
    x->ComputeProgram = clCreateProgramWithSource(x->ComputeContext, 1, (const char **) & source, NULL, &err);
    if (!x->ComputeProgram || err != CL_SUCCESS)
    {
        error("Error: Failed to create compute program! %d\n", err);
        return err;
    }
	
    // Build the program executable
    //
    //if (x->post!=0) post(SEPARATOR);
    if (x->post!=0) post("Building compute program...\n");
    err = clBuildProgram(x->ComputeProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
		
        error("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(x->ComputeProgram, x->ComputeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        error("%s\n", buffer);
        return err;
    }
	
    // Create the compute kernel from within the program
    //
    int i = 0;
    for(i = 0; i < COMPUTE_KERNEL_COUNT; i++)
    {   
        if (x->post!=0) post("Creating kernel '%s'...\n", ComputeKernelMethods[i]);
		
        x->ComputeKernels[i] = clCreateKernel(x->ComputeProgram, ComputeKernelMethods[i], &err);
        if (!x->ComputeKernels[i] || err != CL_SUCCESS)
        {
            error("Error: Failed to create compute kernel!\n");
            return err;
        }
		
        // Get the maximum work group size for executing the kernel on the device
        //
        size_t max = 1;
        err = clGetKernelWorkGroupInfo(x->ComputeKernels[i], x->ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max, NULL);
        if (err != CL_SUCCESS)
        {
            error("Error: Failed to retrieve kernel work group info! %d\n", err);
            return err;
        }
		
        x->ComputeKernelWorkGroupSizes[i] = (max > 1) ? FloorPow2(max) : max;  // use nearest power of two (less than max)
		
        if (x->post!=0) post("%s MaxWorkGroupSize: %d\n", ComputeKernelMethods[i], (int)x->ComputeKernelWorkGroupSizes[i]);
		
    }
	
    return CreateComputeResult(x);
	
}

static int 
SetupComputeDevices(t_jit_cl_noise *x, int gpu)
{
    int err;
	size_t returned_size;
    x->ComputeDeviceType = gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	
	
    // Locate a compute device
    //
    err = clGetDeviceIDs(NULL, x->ComputeDeviceType, 1, &x->ComputeDeviceId, NULL);
    if (err != CL_SUCCESS)
    {
        error("Error: Failed to locate compute device!\n");
        return err;
    }
	
    // Create a context containing the compute device(s)
    //
    x->ComputeContext = clCreateContext(0, 1, &x->ComputeDeviceId, clLogMessagesToStdoutAPPLE, NULL, &err);
    if (!x->ComputeContext)
    {
        error("Error: Failed to create a compute context!\n");
        return err;
    }
	
	
    unsigned int device_count;
    cl_device_id device_ids[16];
	
    err = clGetContextInfo(x->ComputeContext, CL_CONTEXT_DEVICES, sizeof(device_ids), device_ids, &returned_size);
    if(err)
    {
        error("Error: Failed to retrieve compute devices for context!\n");
        return err;
    }
    
    device_count = returned_size / sizeof(cl_device_id);
    
    int i = 0;
    int device_found = 0;
    cl_device_type device_type;	
    for(i = 0; i < device_count; i++) 
    {
        err = clGetDeviceInfo(device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
        if(device_type == x->ComputeDeviceType) 
        {
            x->ComputeDeviceId = device_ids[i];
            device_found = 1;
            break;
        }	
    }
    
    if(!device_found)
    {
        error("Error: Failed to locate compute device!\n");
        return err;
    }
	
    // Create a command queue
    //
    x->ComputeCommands = clCreateCommandQueue(x->ComputeContext, x->ComputeDeviceId, 0, &err);
    if (!x->ComputeCommands)
    {
        error("Error: Failed to create a command queue!\n");
        return err;
    }
	
    // Report the device vendor and device name
    // 
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(x->ComputeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(x->ComputeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    if (err != CL_SUCCESS)
    {
        error("Error: Failed to retrieve device info!\n");
        return err;
    }
	
    //post(SEPARATOR);
    if (x->post!=0) post("Connecting to %s %s...\n", vendor_name, device_name);
	
    return CL_SUCCESS;
}

static void
ShutdownCompute(t_jit_cl_noise *x)
{
    //post(SEPARATOR);
    if (x->post!=0) post("Shutting down...\n");
	
    clFinish(x->ComputeCommands);
    clReleaseKernel(x->ComputeKernel);
    clReleaseProgram(x->ComputeProgram);
    clReleaseCommandQueue(x->ComputeCommands);
    clReleaseMemObject(x->ComputeResult);

}

static int 
clInitialize(t_jit_cl_noise *x,int gpu)
{
    int err;
/*    err = SetupGraphics();
    if (err != GL_NO_ERROR)
    {
        printf ("Failed to setup OpenGL state!");
        exit (err);
    }
*/	
	CreateBuffers(x);
	
    err = SetupComputeDevices(x,gpu);
    if(err != CL_SUCCESS)
    {
        error ("Failed to connect to compute device! Error %d\n", err);
        return (err);
    }
	
    err = SetupComputeKernels(x);
    if (err != CL_SUCCESS)
    {
        error ("Failed to setup compute kernel! Error %d\n", err);
        return (err);
    }
    //post(SEPARATOR);
    if (x->post!=0) post("OpenCL initialized.\n");
	
    return CL_SUCCESS;
}



///////				Jitter				///////				Jitter				///////				Jitter				///////				Jitter				
///////				Jitter				///////				Jitter				///////				Jitter				///////				Jitter				
///////				Jitter				///////				Jitter				///////				Jitter				///////				Jitter				

t_jit_err jit_cl_noise_init(void) 
{
	long attrflags=0;
	t_jit_object *attr,*mop;
	
	// create our class
	_jit_cl_noise_class = jit_class_new("jit_cl_noise",(method)jit_cl_noise_new,(method)jit_cl_noise_free,
		sizeof(t_jit_cl_noise),0L);

	// create a new instance of jit_mop with 1 input, and 1 output
	mop = jit_object_new(_jit_sym_jit_mop,1,1);
	
	// enforce a single type for all inputs and outputs
	jit_mop_single_type(mop,_jit_sym_char);
	
	// enforce a single plane count for all inputs and outputs
	jit_mop_single_planecount(mop,4);	
	
	// add our jit_mop object as an adornment to our class
	jit_class_addadornment(_jit_cl_noise_class,mop);

	// add methods to our class
	jit_class_addmethod(_jit_cl_noise_class, (method)jit_cl_noise_matrix_calc, 		"matrix_calc", 		A_CANT, 0L);

	// add attributes to our class	
	attrflags = JIT_ATTR_GET_DEFER_LOW | JIT_ATTR_SET_USURP_LOW;
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"mode",_jit_sym_long,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,mode));
	jit_attr_addfilterset_clip(attr,0,2,TRUE,TRUE);	//clip to 0-1
	jit_class_addattr(_jit_cl_noise_class,attr);

	attr = jit_object_new(_jit_sym_jit_attr_offset,"post",_jit_sym_long,attrflags,
						  (method)0L,(method)0L,calcoffset(t_jit_cl_noise,post));
	jit_attr_addfilterset_clip(attr,0,1,TRUE,TRUE);	//clip to 0-1
	jit_class_addattr(_jit_cl_noise_class,attr);

	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"scale",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,scale));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"biasX",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,biasX));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"biasY",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,biasY));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"lacunarity",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,lacunarity));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"increment",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,increment));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"octaves",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,octaves));
	jit_class_addattr(_jit_cl_noise_class,attr);
	
	attr = jit_object_new(_jit_sym_jit_attr_offset,"amplitude",_jit_sym_float32,attrflags,
		(method)0L,(method)0L,calcoffset(t_jit_cl_noise,amplitude));
	jit_class_addattr(_jit_cl_noise_class,attr);
	

	// register our class
	jit_class_register(_jit_cl_noise_class);


	
	
	return JIT_ERR_NONE;
}


t_jit_err jit_cl_noise_matrix_calc(t_jit_cl_noise *x, void *inputs, void *outputs)
{
	t_jit_err err=JIT_ERR_NONE;
	long out_savelock;
	t_jit_matrix_info out_minfo;
	char *out_bp;
	long i,dimcount,planecount,dim[JIT_MATRIX_MAX_DIMCOUNT];
	void *out_matrix;
	
	// get the zeroth index input and output from the 
	// corresponding input and output lists
	out_matrix 	= jit_object_method(outputs,_jit_sym_getindex,0);

	// if our object and both our input and output matrices
	// are valid, then process, else return an error
	if (x&&out_matrix) 
	{
		// lock our input and output matrices	
		out_savelock = (long) jit_object_method(out_matrix,_jit_sym_lock,1);

		// fill out our matrix info structs for our input and output			
		jit_object_method(out_matrix,_jit_sym_getinfo,&out_minfo);
		
		// get our matrix data pointers
		jit_object_method(out_matrix,_jit_sym_getdata,&out_bp);
		
		// if our data pointers are invalid, set error, and cleanup
		if (!out_bp) { err=JIT_ERR_INVALID_OUTPUT; goto out;}
		
		// enforce compatible types
		//if ((in_minfo.type!=_jit_sym_char)||(in_minfo.type!=out_minfo.type)) { 
		//	err=JIT_ERR_MISMATCH_TYPE; 
		//	goto out;
		//}		

		// enforce compatible planecount
		if (out_minfo.planecount!=4) { 
			err=JIT_ERR_MISMATCH_PLANE; 
			goto out;
		}		

		// get dimensions/planecount
		dimcount   = out_minfo.dimcount;
		planecount = out_minfo.planecount;			
		for (i=0;i<dimcount;i++) 
		{
			// if input and output are not matched in
			// size, use the intersection of the two
			dim[i] = out_minfo.dim[i];
		}		
		
		// if dims are changed, update openCL buffers
		if ((dim[0]!=x->Width)||(dim[1]!=x->Height))
			{
				x->Width = dim[0];
				x->Height = dim[1];
				
				CreateBuffers(x);
				
				err=CreateComputeResult(x);
				if (err!=0)
					{
						error("Failed to recreate compute result(%d)\n",err);
						goto out;
					}
				
			}
			
		//jit_unpack_calculate_ndim(dimcount, dim, planecount, x->offset[i], &in_minfo, in_bp, out_minfo + i, out_bp[i]);
		
		///// openCL stuff:
		int err;                            // error code returned from api calls
		
		//x->scale = x->gscale;
		
		
		err = Recompute(x);
		if (err != 0)
		{
			error("Error %d from Recompute!\n", err);
			goto out;
			//return err;
		}
		
		if (err == 0)
			memcpy( out_bp, x->HostImageBuffer, x->Width*x->Height*x->TextureTypeSize*4);

	} else {

		return JIT_ERR_INVALID_PTR;
	}
	
out:
	// restore matrix lock state to previous value
	jit_object_method(out_matrix,_jit_sym_lock,out_savelock);
	
	return err;
}


t_jit_cl_noise *jit_cl_noise_new(void)
{
	t_jit_cl_noise *x;
		
	if (x=(t_jit_cl_noise *)jit_object_alloc(_jit_cl_noise_class)) {		
		
		x->post = 1;
		
		// openCL
		x->HostImageBuffer = 0;
		
		x->TextureTypeSize = sizeof(char);
		x->Width = 512;
		x->Height = 512;
		
		x->scale                            = 20.0f;
		x->biasX							= 128.0f;
		x->biasY                            = 128.0f;
		x->lacunarity                       = 2.02f;
		x->increment                        = 1.0f;
		x->octaves                          = 3.3f;
		x->amplitude                        = 1.0f;
	
		x->mode	= 0;
		
		
	} else {
		x = NULL;
	}
	
	// openCL stuff	///	///		///	///			///			///			///			///			///			///			///
	if (x) 	clInitialize(x, 1);
	
	return x;
}

void jit_cl_noise_free(t_jit_cl_noise *x)
{
	//nada

	// OpenCL stuff:
	// Shutdown and cleanup
	ShutdownCompute(x);
  	
}
