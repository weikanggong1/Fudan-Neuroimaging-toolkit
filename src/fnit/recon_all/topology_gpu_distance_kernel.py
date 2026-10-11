"""源有序F32距离核：不同体素独立，面序、分支、剪枝与同分归并保持。"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _dot(x,y,z,a,b,c):return (x*a+y*b)+z*c


@triton.jit
def _load3(data,base):
    return tl.load(data+base),tl.load(data+base+1),tl.load(data+base+2)


@triton.jit
def _edge(ax,ay,az,bx,by,bz,ex,ey,ez,length,pax,pay,paz,pbx,pby,pbz,enx,eny,enz):
    projection=_dot(ax,ay,az,ex,ey,ez)
    start=_dot(ax,ay,az,ax,ay,az);finish=_dot(bx,by,bz,bx,by,bz)
    inner=tl.maximum(0.,start-tl.div_rn(projection*projection,length))
    sq=tl.where(projection<0.,start,tl.where(projection<length,inner,finish))
    sign=tl.where(projection<0.,_dot(pax,pay,paz,ax,ay,az),tl.where(projection<length,_dot(enx,eny,enz,ax,ay,az),_dot(pbx,pby,pbz,bx,by,bz)))
    return sq,sign


@triton.jit
def distance_kernel(Fields,Bounds,Offsets,Rows,Origin,Scale,Output,NX:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0);index=block*BLOCK+tl.arange(0,BLOCK);live=index<NX*NY*NZ
    i=index//(NY*NZ);j=(index//NZ)%NY;k=index%NZ
    ox,oy,oz=_load3(Origin,0);sx,sy,sz=_load3(Scale,0)
    px=ox+tl.div_rn(i.to(tl.float32),sx);py=oy+tl.div_rn(j.to(tl.float32),sy);pz=oz+tl.div_rn(k.to(tl.float32),sz)
    current=tl.full((BLOCK,),1000.,tl.float32)
    start=tl.load(Offsets+block);stop=tl.load(Offsets+block+1)
    for slot in range(start,stop):
        row=tl.load(Rows+slot);base=row*55;b=row*7
        active=live&(i>=tl.load(Bounds+b+1))&(i<=tl.load(Bounds+b+2))&(j>=tl.load(Bounds+b+3))&(j<=tl.load(Bounds+b+4))&(k>=tl.load(Bounds+b+5))&(k<=tl.load(Bounds+b+6))
        if tl.sum(active.to(tl.int32),axis=0)>0:
            cx,cy,cz=_load3(Fields,base+51);far=tl.load(Fields+base+54)
            dx=px-cx;dy=py-cy;dz=pz-cz;center_sq=_dot(dx,dy,dz,dx,dy,dz)
            bound=tl.abs(current)+far
            active=active&~(center_sq*0.98>bound*bound)
            x0,y0,z0=_load3(Fields,base);x1,y1,z1=_load3(Fields,base+3);x2,y2,z2=_load3(Fields,base+6)
            nx,ny,nz=_load3(Fields,base+9)
            ax,ay,az=px-x0,py-y0,pz-z0;bx,by,bz=px-x1,py-y1,pz-z1;cx,cy,cz=px-x2,py-y2,pz-z2
            h0x,h0y,h0z=_load3(Fields,base+30);h1x,h1y,h1z=_load3(Fields,base+33);h2x,h2y,h2z=_load3(Fields,base+36)
            val0=_dot(ax,ay,az,h0x,h0y,h0z);val1=_dot(bx,by,bz,h1x,h1y,h1z);val2=_dot(cx,cy,cz,h2x,h2y,h2z)
            plane=(val0>=0.)&(val1>=0.)&(val2>=0.)
            mx=((ax+bx)+cx)*0.3333333;my=((ay+by)+cy)*0.3333333;mz=((az+bz)+cz)*0.3333333
            candidate=_dot(nx,ny,nz,mx,my,mz)
            p0x,p0y,p0z=_load3(Fields,base+12);p1x,p1y,p1z=_load3(Fields,base+15);p2x,p2y,p2z=_load3(Fields,base+18)
            least=tl.full((BLOCK,),1000000.,tl.float32);sign=tl.full((BLOCK,),0.,tl.float32)
            e0x,e0y,e0z=_load3(Fields,base+21);n0x,n0y,n0z=_load3(Fields,base+39)
            sq,sg=_edge(ax,ay,az,bx,by,bz,e0x,e0y,e0z,tl.load(Fields+base+48),p0x,p0y,p0z,p1x,p1y,p1z,n0x,n0y,n0z)
            use=val0<=0.;least=tl.where(use,sq,least);sign=tl.where(use,sg,sign)
            e1x,e1y,e1z=_load3(Fields,base+24);n1x,n1y,n1z=_load3(Fields,base+42)
            sq,sg=_edge(bx,by,bz,cx,cy,cz,e1x,e1y,e1z,tl.load(Fields+base+49),p1x,p1y,p1z,p2x,p2y,p2z,n1x,n1y,n1z)
            use=(val1<=0.)&(least>sq);least=tl.where(use,sq,least);sign=tl.where(use,sg,sign)
            e2x,e2y,e2z=_load3(Fields,base+27);n2x,n2y,n2z=_load3(Fields,base+45)
            sq,sg=_edge(cx,cy,cz,ax,ay,az,e2x,e2y,e2z,tl.load(Fields+base+50),p2x,p2y,p2z,p0x,p0y,p0z,n2x,n2y,n2z)
            use=(val2<=0.)&(least>sq);least=tl.where(use,sq,least);sign=tl.where(use,sg,sign)
            positive=libdevice.sqrt_rn(least);signed=tl.where(sign>0.,positive,-positive)
            edge_valid=~((current*current)*1.01<least)&~(positive>tl.abs(current))
            candidate=tl.where(plane,candidate,signed)
            update=active&(plane|edge_valid)&((tl.abs(candidate)<tl.abs(current))|((tl.abs(candidate)==tl.abs(current))&(candidate>0.)))
            current=tl.where(update,candidate,current)
    tl.store(Output+index,current,mask=live)
