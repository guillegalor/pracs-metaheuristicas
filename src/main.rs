extern crate csv;

use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::process;

struct TextureRecord {
    id: i32,
    data: [f32; 40],
    class: i32,
}

impl TextureRecord {
    fn new() -> TextureRecord {
        TextureRecord {
            id: -1,
            data: [0.0; 40],
            class: -1,
        }
    }

    fn euclidian_distance(&self, other: &TextureRecord) -> f32 {
        let mut dist = 0.0;
        for it in 0..40 {
            dist += (self.data[it] - other.data[it]) * (self.data[it] - other.data[it]);
        }
        dist = dist.sqrt();

        return dist;
    }
}

fn make_partitions(data: Vec<TextureRecord>) -> Vec<Vec<TextureRecord>> {
    let folds = 5;

    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<TextureRecord>> = Vec::new();

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

fn classifier_1nn(data: Vec<TextureRecord>, item: TextureRecord) -> i32 {
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
    let mut data: Vec<TextureRecord> = Vec::new();
    let mut rdr = csv::Reader::from_path("data/texture.csv")?;

    for result in rdr.records() {
        let mut aux_record = TextureRecord::new();
        let record = result?;
        let mut counter = 0;

        for field in record.iter() {
            // CSV structure: id , ... 40 data ... , class
            if counter == 0 {
                aux_record.id = field.parse::<i32>().unwrap();
            } else if counter != 41 {
                aux_record.data[counter - 1] = field.parse::<f32>().unwrap();
            } else {
                aux_record.class = field.parse::<i32>().unwrap();
            }

            counter += 1;
        }

        data.push(aux_record);
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
