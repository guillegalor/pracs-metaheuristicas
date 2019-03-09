extern crate csv;
#[macro_use]
extern crate serde_derive;

use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::process;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct Texture {
    id: f32,
    A1: f32,
    A2: f32,
    A3: f32,
    A4: f32,
    A5: f32,
    A6: f32,
    A7: f32,
    A8: f32,
    A9: f32,
    A10: f32,
    A11: f32,
    A12: f32,
    A13: f32,
    A14: f32,
    A15: f32,
    A16: f32,
    A17: f32,
    A18: f32,
    A19: f32,
    A20: f32,
    A21: f32,
    A22: f32,
    A23: f32,
    A24: f32,
    A25: f32,
    A26: f32,
    A27: f32,
    A28: f32,
    A29: f32,
    A30: f32,
    A31: f32,
    A32: f32,
    A33: f32,
    A34: f32,
    A35: f32,
    A36: f32,
    A37: f32,
    A38: f32,
    A39: f32,
    A40: f32,
    class: i32,
}

impl Texture {
    fn euclidian_distance(&self, other: &Texture) -> f32 {
        return ((self.A1 - other.A1) * (self.A1 - other.A1)
            + (self.A2 - other.A2) * (self.A2 - other.A2)
            + (self.A3 - other.A3) * (self.A3 - other.A3)
            + (self.A4 - other.A4) * (self.A4 - other.A4)
            + (self.A5 - other.A5) * (self.A5 - other.A5)
            + (self.A6 - other.A6) * (self.A6 - other.A6)
            + (self.A7 - other.A7) * (self.A7 - other.A7)
            + (self.A8 - other.A8) * (self.A8 - other.A8)
            + (self.A9 - other.A9) * (self.A9 - other.A9)
            + (self.A10 - other.A10) * (self.A10 - other.A10)
            + (self.A11 - other.A11) * (self.A11 - other.A11)
            + (self.A12 - other.A12) * (self.A12 - other.A12)
            + (self.A13 - other.A13) * (self.A13 - other.A13)
            + (self.A14 - other.A14) * (self.A14 - other.A14)
            + (self.A15 - other.A15) * (self.A15 - other.A15)
            + (self.A16 - other.A16) * (self.A16 - other.A16)
            + (self.A17 - other.A17) * (self.A17 - other.A17)
            + (self.A18 - other.A18) * (self.A18 - other.A18)
            + (self.A19 - other.A19) * (self.A19 - other.A19)
            + (self.A20 - other.A20) * (self.A20 - other.A20)
            + (self.A21 - other.A21) * (self.A21 - other.A21)
            + (self.A22 - other.A22) * (self.A22 - other.A22)
            + (self.A23 - other.A23) * (self.A23 - other.A23)
            + (self.A24 - other.A24) * (self.A24 - other.A24)
            + (self.A25 - other.A25) * (self.A25 - other.A25)
            + (self.A26 - other.A26) * (self.A26 - other.A26)
            + (self.A27 - other.A27) * (self.A27 - other.A27)
            + (self.A28 - other.A28) * (self.A28 - other.A28)
            + (self.A29 - other.A29) * (self.A29 - other.A29)
            + (self.A30 - other.A30) * (self.A30 - other.A30)
            + (self.A31 - other.A31) * (self.A31 - other.A31)
            + (self.A32 - other.A32) * (self.A32 - other.A32)
            + (self.A33 - other.A33) * (self.A33 - other.A33)
            + (self.A34 - other.A34) * (self.A34 - other.A34)
            + (self.A35 - other.A35) * (self.A35 - other.A35)
            + (self.A36 - other.A36) * (self.A36 - other.A36)
            + (self.A37 - other.A37) * (self.A37 - other.A37)
            + (self.A38 - other.A38) * (self.A38 - other.A38)
            + (self.A39 - other.A39) * (self.A39 - other.A39)
            + (self.A40 - other.A40) * (self.A40 - other.A40))
            .sqrt();
    }
}

fn make_partitions(data: Vec<Texture>) -> Vec<Vec<Texture>> {
    let folds = 5;

    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<Texture>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data {
        let counter = categories_count.entry(example.class).or_insert(0);
        partitions[*counter].push(example);

        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

fn classifier_1nn(data: Vec<Texture>, item: Texture) -> i32 {
    let mut c_min = data[0].class;
    let mut d_min = item.euclidian_distance(&data[0]);

    for example in data {
        let distance = item.euclidian_distance(&example);
        if distance < d_min {
            c_min = example.class;
            d_min = distance;
        }
    }

    return c_min;
}

fn run() -> Result<(), Box<Error>> {
    let mut data: Vec<Texture> = Vec::new();
    let mut rdr = csv::Reader::from_path("data/texture.csv")?;
    for result in rdr.deserialize() {
        let record: Texture = result?;
        data.push(record);
    }

    let partitions = make_partitions(data);

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("error: {}", err);
        process::exit(1);
    }
}
