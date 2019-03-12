extern crate csv;

use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::num;
use std::process;

#[derive(Copy, Clone)]
struct TextureRecord {
    id: i32,
    attributes: [f32; 40],
    class: i32,
}

impl TextureRecord {
    fn new() -> TextureRecord {
        TextureRecord {
            id: -1,
            attributes: [0.0; 40],
            class: -1,
        }
    }

    fn euclidian_distance(&self, other: &TextureRecord) -> f32 {
        let mut dist = 0.0;
        for it in 0..40 {
            dist += (self.attributes[it] - other.attributes[it])
                * (self.attributes[it] - other.attributes[it]);
        }
        dist = dist.sqrt();

        return dist;
    }

    // TODO Check if weights length coincide with the num of attributes
    // Euclidian distance considering weights
    fn eu_dist_with_weigths(&self, other: &TextureRecord, weights: &Vec<f32>) -> f32 {
        let mut dist = 0.0;
        for attr in 0..40 {
            dist += (self.attributes[attr] - other.attributes[attr])
                * (self.attributes[attr] - other.attributes[attr])
                * weights[attr];
        }
        dist = dist.sqrt();

        return dist;
    }
}

fn normalize_data(data: Vec<TextureRecord>) -> Vec<TextureRecord> {
    let mut mins = [std::f32::MAX; 40];
    let mut maxs = [std::f32::MIN; 40];

    for elem in data.iter() {
        for attr in 0..40 {
            // Calculates min
            if elem.attributes[attr] < mins[attr] {
                mins[attr] = elem.attributes[attr];
            }
            // Calculates max
            if elem.attributes[attr] > mins[attr] {
                maxs[attr] = elem.attributes[attr];
            }
        }
    }

    let mut new_data = data;

    let mut max_distances: [f32; 40] = [0.0; 40];
    for attr in 0..40 {
        max_distances[attr] = maxs[attr] - mins[attr];
    }

    for elem in new_data.iter_mut() {
        for attr in 0..40 {
            elem.attributes[attr] = (elem.attributes[attr] - mins[attr]) / max_distances[attr];
        }
    }

    return new_data;
}

fn make_partitions(data: &Vec<TextureRecord>) -> Vec<Vec<TextureRecord>> {
    let folds = 5;

    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<TextureRecord>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data.iter() {
        let counter = categories_count.entry(example.class).or_insert(0);
        partitions[*counter].push(example.clone());

        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

fn classifier_1nn(data: &Vec<TextureRecord>, item: &TextureRecord) -> i32 {
    let mut c_min = data[0].class;
    let mut d_min = item.euclidian_distance(&data[0]);

    for example in data {
        if example.id != item.id {
            let distance = item.euclidian_distance(&example);
            if distance < d_min {
                c_min = example.class;
                d_min = distance;
            }
        }
    }

    return c_min;
}

fn classifier_1nn_with_weights(
    data: &Vec<TextureRecord>,
    item: &TextureRecord,
    weights: &Vec<f32>,
) -> i32 {
    let mut c_min = data[0].class;
    let mut d_min = item.euclidian_distance(&data[0]);

    for example in data {
        if example.id != item.id {
            let distance = item.eu_dist_with_weigths(&example, weights);
            if distance < d_min {
                c_min = example.class;
                d_min = distance;
            }
        }
    }

    return c_min;
}

// TODO Change magic num "40"
// Greedy algorithm
fn relief_algorithm(data: &Vec<TextureRecord>) -> Vec<f32> {
    let mut weights = Vec::with_capacity(40);
    for elem in data.iter() {
        let mut nearest_enemy = TextureRecord::new();
        let mut nearest_ally = TextureRecord::new();
        let mut best_enemy_distance = std::f32::MAX;
        let mut best_ally_distance = std::f32::MAX;

        for candidate in data.iter() {
            if elem.id != candidate.id {
                let aux_distance = elem.euclidian_distance(candidate);

                // Ally search
                if elem.class == candidate.class {
                    if aux_distance < best_ally_distance {
                        best_ally_distance = aux_distance;
                        nearest_ally = candidate.clone();
                    }
                }
                // Enemy search
                else {
                    if aux_distance < best_enemy_distance {
                        best_enemy_distance = aux_distance;
                        nearest_enemy = candidate.clone();
                    }
                }
            }
        }

        for attribute in 0..40 {
            let attr_ally_dist =
                (elem.attributes[attribute] - nearest_ally.attributes[attribute]).abs();
            let attr_enemy_dist =
                (elem.attributes[attribute] - nearest_enemy.attributes[attribute]).abs();
            weights[attribute] = weights[attribute] + attr_enemy_dist - attr_ally_dist;
        }
    }

    return weights;
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
                aux_record.attributes[counter - 1] = field.parse::<f32>().unwrap();
            } else {
                aux_record.class = field.parse::<i32>().unwrap();
            }

            counter += 1;
        }

        data.push(aux_record);
    }

    data = normalize_data(data);

    let partitions = make_partitions(&data);

    // Stablish training and test sets
    for test in 0..5 {
        let mut training_set: Vec<TextureRecord> = Vec::new();
        let mut test_set: Vec<TextureRecord> = Vec::new();

        for part in 0..5 {
            if part != test {
                training_set.extend(&partitions[part]);
            } else {
                test_set = partitions[part].clone();
            }
        }

        // 1-NN
        let mut guessin_1nn: Vec<i32> = Vec::new();

        for elem in test_set.iter() {
            guessin_1nn.push(classifier_1nn(&training_set, elem));
        }

        // Relief algorithm (greedy)
        let relief_weights = relief_algorithm(&training_set);
        let mut guessin_relief: Vec<i32> = Vec::new();

        for elem in test_set.iter() {
            guessin_relief.push(classifier_1nn_with_weights(
                &training_set,
                elem,
                &relief_weights,
            ));
        }
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("error: {}", err);
        process::exit(1);
    }
}
