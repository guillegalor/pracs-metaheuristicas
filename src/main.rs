extern crate csv;
#[macro_use]
extern crate prettytable;

use prettytable::{Cell, Row, Table};

use rand::distributions::{Distribution, Normal, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;

use std::collections::HashMap;
use std::error::Error;
use std::process;
use std::time::Instant;

// ---------------------------------------------------------------------------------
// DataElem definitions and specific functions for it
// ---------------------------------------------------------------------------------

/// Trait for CSV data
///
/// **Note**: The struct implementing this trait must also implement `Copy` and `Clone`.

pub trait DataElem<T> {
    fn new() -> T;
    fn get_num_attributes() -> usize;
    fn get_id(&self) -> i32;
    fn get_class(&self) -> i32;
    fn get_attribute(&self, index: usize) -> f32;
    fn set_id(&mut self, id: i32);
    fn set_class(&mut self, class: i32);
    fn set_attribute(&mut self, index: usize, attr: f32);
}

/// Calculate the "euclidian_distance" distance between two `T` elements
/// asigning a weight to every attribute
///
/// # Arguments
///
/// * `one` - First `T` element
/// * `other` - Second `T` element
/// * `weights` - Vec with a weight(between 0 and 1) for every attribute
///
/// # Returns
/// A `Result ` with:
/// * `Err(&'static str)` if the number or weights doesn't coincide with the number ot attrs of `T`
/// * `Ok(f32)`
pub fn eu_dist_with_weigths<T: DataElem<T> + Copy + Clone>(
    one: &T,
    other: &T,
    weights: &Vec<f32>,
) -> Result<f32, &'static str> {
    if T::get_num_attributes() != weights.len() {
        return Err("El numero de pesos no coincide con el numero de atributos");
    }

    let mut dist = 0.0;
    for attr in 0..T::get_num_attributes() {
        dist += (one.get_attribute(attr) - other.get_attribute(attr))
            * (one.get_attribute(attr) - other.get_attribute(attr))
            * weights[attr];
    }
    dist = dist.sqrt();

    return Ok(dist);
}

#[derive(Copy, Clone)]
struct TextureRecord {
    id: i32,
    attributes: [f32; 40],
    class: i32,
}

impl DataElem<TextureRecord> for TextureRecord {
    fn new() -> TextureRecord {
        TextureRecord {
            id: -1,
            attributes: [0.0; 40],
            class: -1,
        }
    }

    fn get_num_attributes() -> usize {
        return 40;
    }

    fn get_id(&self) -> i32 {
        return self.id;
    }

    fn get_class(&self) -> i32 {
        return self.class;
    }

    // NOTE Not sure if we should check if index is in range
    fn get_attribute(&self, index: usize) -> f32 {
        return self.attributes[index];
    }

    fn set_id(&mut self, id: i32) {
        self.id = id;
    }

    fn set_class(&mut self, class: i32) {
        self.class = class;
    }

    fn set_attribute(&mut self, index: usize, attr: f32) {
        self.attributes[index] = attr;
    }
}

//---------------------------------------------------------------------------------------
// Auxiliary functions to treat data set
//---------------------------------------------------------------------------------------

/// Normalize attributes of a vec of `T` so everyone is in [0,1]
/// # Arguments
///
/// * `data` - Vec of `T` being normalized
///
/// # Returns
/// Returns a vec of `T` normalized

pub fn normalize_data<T: DataElem<T> + Copy + Clone>(data: Vec<T>) -> Vec<T> {
    let num_attrs = T::get_num_attributes();

    let mut mins = vec![std::f32::MAX; num_attrs];
    let mut maxs = vec![std::f32::MIN; num_attrs];

    for elem in data.iter() {
        for attr in 0..num_attrs {
            // Calculates min
            if elem.get_attribute(attr) < mins[attr] {
                mins[attr] = elem.get_attribute(attr);
            }
            // Calculates max
            if elem.get_attribute(attr) > maxs[attr] {
                maxs[attr] = elem.get_attribute(attr);
            }
        }
    }

    let mut new_data = data;

    let mut max_distances = vec![0.0; num_attrs];
    for attr in 0..num_attrs {
        max_distances[attr] = maxs[attr] - mins[attr];
    }

    for elem in new_data.iter_mut() {
        for attr in 0..num_attrs {
            elem.set_attribute(
                attr,
                (elem.get_attribute(attr) - mins[attr]) / max_distances[attr],
            );
        }
    }

    return new_data;
}

/// Makes partitions keeping the class diversity.
///
/// # Arguments
///
/// * `data` - Vec with all the data.
///
/// # Returns
/// Returns a vector with 5 vectors of `T`.
pub fn make_partitions<T: DataElem<T> + Copy + Clone>(data: &Vec<T>) -> Vec<Vec<T>> {
    let folds = 5;

    // Hash tab that has an entry for every diferent class, and tells where should be
    // the next elemt
    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<T>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data.iter() {
        // If that class was already found, gets the partition that this elem should go
        // If not, insert a new entry for that class with value 0 (first partition)
        let counter = categories_count.entry(example.get_class()).or_insert(0);

        partitions[*counter].push(example.clone());

        // The next elem with same class as elem will go to the next partition
        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

//---------------------------------------------------------------------------------------
// Core functions for algorithms
//---------------------------------------------------------------------------------------

/// Classifies one `T` item using a Vec of `T` elems, a Vec of weights and the
/// 1nn algorithm
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
/// * `item` - `T` elem being classified
/// * `weights` - Vec of f32 representing weights
///
/// # Returns
/// A `Result` with:
/// * `Err(&'static str)` if the number or weights doesn't coincide with the number ot attrs of `T`
/// * `Ok(i32)`

pub fn classifier_1nn_with_weights<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    item: &T,
    weights: &Vec<f32>,
) -> Result<i32, Box<Error>> {
    let mut c_min = data[0].get_class();
    let mut d_min = eu_dist_with_weigths(item, &data[0], weights)?;
    let weights_discarding: Vec<f32> = weights
        .iter()
        .map(|x| if *x < 0.2 { 0.0 } else { *x })
        .collect();

    for example in data.iter().skip(1) {
        if example.get_id() != item.get_id() {
            let aux_distance = eu_dist_with_weigths(item, example, &weights_discarding)?;
            if aux_distance < d_min {
                c_min = example.get_class();
                d_min = aux_distance;
            }
        }
    }

    return Ok(c_min);
}

/// Calculates a vec of weights using a vec of `T` elems as knowledge
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
///
/// # Returns
/// A Vec with the same num of f32 elems as the num of attributes of `T`

pub fn relief_algorithm<T: DataElem<T> + Copy + Clone>(data: &Vec<T>) -> Vec<f32> {
    let num_attrs = T::get_num_attributes();
    let mut weights = vec![0.0; num_attrs];
    let weights_eu_dist = vec![1.; num_attrs];

    for elem in data.iter() {
        let mut nearest_enemy_index = 0;
        let mut nearest_ally_index = 0;
        let mut best_enemy_distance = std::f32::MAX;
        let mut best_ally_distance = std::f32::MAX;

        for (index, candidate) in data.iter().enumerate() {
            if elem.get_id() != candidate.get_id() {
                let aux_distance = eu_dist_with_weigths(elem, candidate, &weights_eu_dist)
                    .expect("Euclidian distance with weights relief algorithm");

                // Ally search
                if elem.get_class() == candidate.get_class() {
                    if aux_distance < best_ally_distance {
                        best_ally_distance = aux_distance;
                        nearest_ally_index = index;
                    }
                }
                // Enemy search
                else {
                    if aux_distance < best_enemy_distance {
                        best_enemy_distance = aux_distance;
                        nearest_enemy_index = index;
                    }
                }
            }
        }

        let nearest_ally = data[nearest_ally_index].clone();
        let nearest_enemy = data[nearest_enemy_index].clone();

        for attr in 0..num_attrs {
            let attr_ally_dist =
                (elem.get_attribute(attr) - nearest_ally.get_attribute(attr)).abs();
            let attr_enemy_dist =
                (elem.get_attribute(attr) - nearest_enemy.get_attribute(attr)).abs();
            weights[attr] += attr_enemy_dist - attr_ally_dist;
        }
    }

    let mut max_weight = weights[0];

    for w in weights.iter() {
        if *w > max_weight {
            max_weight = *w;
        }
    }

    for w in weights.iter_mut() {
        if *w < 0.0 {
            *w = 0.0;
        } else {
            *w = *w / max_weight;
        }
    }

    return weights;
}

// Local search
// Return the weights vec and the evaluation rate
pub fn local_search<T: DataElem<T> + Copy + Clone>(data: &Vec<T>) -> Vec<f32> {
    let num_attrs = T::get_num_attributes();
    let mut rng = thread_rng();

    // Initialize vector of indexes and shuffles it
    let mut indexes: Vec<usize> = (0..num_attrs).collect();
    indexes.shuffle(&mut rng);

    // Local search parameters
    let mut num_of_mutations = 0;
    let max_mutations = 15000;
    let mut neighbours_without_mutting = 0;
    let max_neighbour_without_muting = 20 * num_attrs;

    // Normal distribution with mean = 0.0, standard deviation = 0.3
    let normal = Normal::new(0.0, 0.3);
    let uniform = Uniform::new(0.0, 1.0);

    // Initialize random weights (using normal distribution)
    let mut weights: Vec<f32> = Vec::with_capacity(T::get_num_attributes());
    for _ in 0..T::get_num_attributes() {
        weights.push(uniform.sample(&mut rng) as f32);
    }

    let mut initial_guessing: Vec<i32> = Vec::new();
    // Initialize current guessing
    for elem in data.iter() {
        initial_guessing.push(
            classifier_1nn_with_weights(data, elem, &weights)
                .expect("No coincide el número de pesos con el de atributos (initial_guessing)"),
        );
    }
    let mut current_ev_rate = evaluation_function(
        class_rate(data, &initial_guessing)
            .expect("No coincide el número de elementos con el número de <<guessings>"),
        red_rate(&weights),
        0.5,
    );

    while neighbours_without_mutting < max_neighbour_without_muting
        && num_of_mutations < max_mutations
    {
        let mut aux_weights = weights.clone();

        if indexes.is_empty() {
            indexes = (0..num_attrs).collect();
            indexes.shuffle(&mut rng);
        }

        let index = indexes.pop().expect("El vector está vacio");

        aux_weights[index] += normal.sample(&mut rng) as f32;
        // Truncate into [0,1]
        if aux_weights[index] < 0. {
            aux_weights[index] = 0.;
        } else if aux_weights[index] > 1. {
            aux_weights[index] = 1.;
        }

        let mut aux_guessing: Vec<i32> = Vec::new();
        // Initialize candidate guessing
        for elem in data.iter() {
            aux_guessing.push(
                classifier_1nn_with_weights(data, elem, &aux_weights)
                    .expect("No coincide el número de pesos  con el de atributos(aux_guessing)"),
            );
        }
        let aux_ev_rate = evaluation_function(
            class_rate(data, &aux_guessing)
                .expect("No coincide el número de elementos con el número de <<guessings>"),
            red_rate(&aux_weights),
            0.5,
        );

        if aux_ev_rate > current_ev_rate {
            current_ev_rate = aux_ev_rate;
            weights = aux_weights;

            neighbours_without_mutting = 0;

            // Refreshes indexes
            indexes = (0..num_attrs).collect();
            indexes.shuffle(&mut rng);
        } else {
            neighbours_without_mutting += 1;
        }

        num_of_mutations += 1;
    }

    return weights;
}

//---------------------------------------------------------------------------------------
// Evaluation function for results
//---------------------------------------------------------------------------------------

/// Calculates the percentage of correctly classified elems
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
/// * `guessing` - Vec of i32 which represents the guessed class for every elem of data
///
/// # Returns
/// A `Result` with:
/// * `Err(&'static str)` if the size of `guessing` doesn't coincide with the size of `data`
/// * `Ok(f32)` with the percentage

pub fn class_rate<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    guessing: &Vec<i32>,
) -> Result<f32, &'static str> {
    if data.len() != guessing.len() {
        return Err("El numero de clases no coincide con el numero de elementos");
    }

    let mut counter = 0.0;

    for (index, elem) in data.iter().enumerate() {
        if elem.get_class() == guessing[index] {
            counter += 1.0;
        }
    }

    return Ok(100.0 * counter / (data.len() as f32));
}

/// Calculates the percentage of attributes that are not going to be used because
/// its weights are too low (under 0.2)
///
/// # Arguments
///
/// * `weights` - Vec of f32 representing weights
///
/// # Returns
/// A f32 with the percentage

pub fn red_rate(weights: &Vec<f32>) -> f32 {
    let mut counter = 0.0;

    for w in weights.iter() {
        if *w < 0.2 {
            counter += 1.0;
        }
    }

    return 100.0 * counter / (weights.len() as f32);
}

/// Calculates the evaluation function using classification and reduction rates
/// previously calculated
/// # Arguments
///
/// * `class_rate` - Classification rate
/// * `red_rate` - Reduction rate
/// * `alpha` - Value of classification importance over reduction (between 0 and 1)
///
/// # Returns
/// An f32 with the result

pub fn evaluation_function(class_rate: f32, red_rate: f32, alpha: f32) -> f32 {
    return alpha * class_rate + (1.0 - alpha) * red_rate;
}

//---------------------------------------------------------------------------------------
// Full algorithms execution functions
//---------------------------------------------------------------------------------------

// 1nn -----------------
pub fn alg_1nn<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
) -> (f32, f32, f32) {
    let mut guessin_1nn: Vec<i32> = Vec::new();
    let weights_eu_dist = vec![1.; T::get_num_attributes()];

    for elem in test_set.iter() {
        guessin_1nn.push(
            classifier_1nn_with_weights(training_set, elem, &weights_eu_dist)
                .expect("Classifier 1nn en alg_1nn"),
        );
    }

    // Results
    let c_rate: f32 = class_rate(test_set, &guessin_1nn)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = 0.0; // (0 cause all weights are equal to 1)
    let ev_rate: f32 = evaluation_function(c_rate, r_rate, 0.5);

    return (c_rate, r_rate, ev_rate);
}

// Relief --------------
pub fn alg_relief<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
) -> (f32, f32, f32) {
    let relief_weights = relief_algorithm(training_set);
    let mut guessin_relief: Vec<i32> = Vec::new();

    for elem in test_set.iter() {
        guessin_relief.push(
            classifier_1nn_with_weights(&training_set, elem, &relief_weights)
                .expect("No coincide el número de pesos  con el de atributos"),
        );
    }

    let c_rate: f32 = class_rate(test_set, &guessin_relief)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = red_rate(&relief_weights);
    let ev_rate: f32 = evaluation_function(c_rate, r_rate, 0.5);

    return (c_rate, r_rate, ev_rate);
}

// Local Search --------
pub fn alg_local_search<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
) -> (f32, f32, f32) {
    let weights = local_search(training_set);

    let mut guessing: Vec<i32> = Vec::new();

    for elem in test_set.iter() {
        guessing.push(
            classifier_1nn_with_weights(&training_set, elem, &weights)
                .expect("No coincide el número de pesos  con el de atributos"),
        );
    }

    let c_rate: f32 = class_rate(test_set, &guessing)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = red_rate(&weights);
    let ev_rate = evaluation_function(c_rate, r_rate, 0.5);

    return (c_rate, r_rate, ev_rate);
}

// ------------------------------------------------------------------------------------

/// Main function
fn run() -> Result<(), Box<Error>> {
    let mut data: Vec<TextureRecord> = Vec::new();
    let mut rdr = csv::Reader::from_path("data/texture.csv")?;

    let mut current_id = 0;
    for result in rdr.records() {
        let mut aux_record = TextureRecord::new();
        let record = result?;

        let mut counter = 0;

        aux_record.id = current_id;

        for field in record.iter() {
            // CSV structure: ... 40 data ... , class
            if counter != TextureRecord::get_num_attributes() {
                aux_record.attributes[counter] = field.parse::<f32>().unwrap();
            } else {
                aux_record.class = field.parse::<i32>().unwrap();
            }

            counter += 1;
        }

        current_id += 1;

        data.push(aux_record);
    }

    data = normalize_data(data);

    let partitions = make_partitions(&data);

    // Output tables declaration
    let mut table_1nn = table!(["Partición", "Tasa_clas", "Tasa_red", "Agregado", "Tiempo"]);
    let mut table_relief = table!(["Partición", "Tasa_clas", "Tasa_red", "Agregado", "Tiempo"]);
    let mut table_localsearch =
        table!(["Partición", "Tasa_clas", "Tasa_red", "Agregado", "Tiempo"]);

    for test in 0..5 {
        // Stablish training and test sets
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
        let mut now = Instant::now();

        let results_1nn = alg_1nn(&training_set, &test_set);

        let time_elapsed_1nn = now.elapsed().as_millis();

        table_1nn.add_row(row![
            test,
            results_1nn.0,
            results_1nn.1,
            results_1nn.2,
            time_elapsed_1nn
        ]);

        // Relief algorithm (greedy)
        now = Instant::now();

        let results_relief = alg_relief(&training_set, &test_set);

        let time_elapsed_relief = now.elapsed().as_millis();

        table_relief.add_row(row![
            test,
            results_relief.0,
            results_relief.1,
            results_relief.2,
            time_elapsed_relief
        ]);

        // Local search algorithm
        now = Instant::now();

        let results_local_search = alg_local_search(&training_set, &test_set);

        let time_elapsed_local_search = now.elapsed().as_millis();

        table_localsearch.add_row(row![
            test,
            results_local_search.0,
            results_local_search.1,
            results_local_search.2,
            time_elapsed_local_search
        ]);
    }

    println!(" Resultados utilizando 1nn");
    table_1nn.printstd();
    println!(" Resultados utilizando Relief");
    table_relief.printstd();
    println!(" Resultados utilizando Busqueda Local");
    table_localsearch.printstd();

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("error: {}", err);
        process::exit(1);
    }
}
