import taichi as ti

#选择使用的设备
ti.init(debug = True) #打开调试模式
#张量：高维数组
#var 标量
#vector 向量
#matrix 矩阵（和张量需要完全区分）
#通过[i,j,k]访问元素
a = ti.var(dt=ti.f32,shape = (42,63)) #标量的张量（张量的每个元素是标量）
b = ti.Vector(3,dt=ti.f32,shape=4) #向量的张量（张量的每个元素是向量）3表示向量的元素，shape是张量的维度
c = ti.Matrix(2,2,dt=ti.f32,shape=(3,5)) #矩阵的张量

loss = ti.var(dt=ti.f32,shape=()) #0D的张量，是一个标量

a[3,4] = 1
print("a[3,4]",a[3,4])

b[2] = [6,7,8]
print("b[0]=",b[0][0],b[0][1],b[0][2])

loss[None] = 3
print("loss[None]",loss[None])

#太极中的kernal：用于计算的函数，可以从python调用
@ti.kernel#实时编译，以block为单位，可并行
def hello() -> ti.i32:#返回值类型
    a = 40
    print(a)
    s = 0
    return s

hello()

#@ti.func #太极的func可以被kernal调用但是不能被python调用
#a/b 永远是浮点数结果
#a//b 则可以返回整数除

#并行for循环
#range for loop
#for x in range(10)只有最外层的循环会自动并行
#for i ,j,k in ti.ndrange((3,8),(2,5),9):
#struct-for loop
n=320
pixel = ti.var(dt=ti.f32,shape=(n*2,n))
@ti.kernel
def paint(t:ti.f32):
    for i,j in pixel:   #遍历tensor的下标，并行执行
        pixel[i,j] = i+j;
        #如果是稀疏矩阵，则遍历只会访问激活的那些部分


#原子操作
#augmented assignment都是原子操作 +=
#ti.atomic_add


#reset
ti.reset()