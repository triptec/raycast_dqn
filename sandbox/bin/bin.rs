use tch::Tensor;
use tch::kind::FLOAT_CPU;
use tch::kind::INT64_CPU;

pub fn main() {
    //let index = Tensor::of_slice(&[2 as i64,2,2,2,2]);
    let index = Tensor::ones(&[5,1], INT64_CPU);
    //dbg!(Vec<i64>::from_tensor(index));
    &index.squeeze().print();
    let a1 = Tensor::zeros(&[5, 5], FLOAT_CPU);
    let a2 = Tensor::ones(&[5,1], FLOAT_CPU);
    let res = a1.index_fill(1, &index, 1.0);
    dbg!(&index, &a1,a2, &res);
    &res.print();
}
