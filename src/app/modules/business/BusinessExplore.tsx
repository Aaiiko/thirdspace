"use client"

import { Business } from "@/types/BusinessTypes";
import React, { useState, useRef, useEffect } from "react";
import { BusinessCard } from "./BusinessCard";
import CheckIcon from "@mui/icons-material/Check";
import ClearIcon from "@mui/icons-material/Clear";
import { database } from "@../../../firebaseConfig";
import { ref, set, get, child } from "firebase/database";
import { useMediaQuery } from "@mui/material";


interface BusinessExploreProps {
    /** The collection of business for the user to iterate through */
    businesses: Business[];
}

export const BusinessExplore = (props: BusinessExploreProps) => {
    const isDesktop = useMediaQuery('(min-width: 1024px)');
    const [currentB, setCurrentB] = useState<number>(0);
    const [rejected, setRejected] = useState<Business[]>([]);
    const [loved, setLoved] = useState<Business[]>([]);
    const [businessArray, setBusinessArray] = useState<Business[]>(props.businesses);
    const touchStartX = useRef<number>(0);
    const touchEndX = useRef<number>(0);

    // useEffect(() => {
    //     if (currentB === props.businesses.length - 1) {
    //         return;
    //     }
    //     fetchBusinesses();
    // }, []);

    const fetchBusinesses = async () => {
        const url = '/test/api';
        console.log("fetching businesses...");
        try{
            const response = await fetch(url);
            const data = await response.json();
            console.log(data);
            if (data) {
                setBusinessArray(data);
                setCurrentB(0);
            }
        } catch (error) {
            console.log("Error fetching businesses: ", error);
        }
    }

    /** Function to save loved business to Firebase */
    const saveLovedBusiness = (business: Business) => {
        set(ref(database, 'loved/' + business.id), business);
    };

    /** Function to save rejected business to Firebase */
    const saveRejectedBusiness = (business: Business) => {
        set(ref(database, 'rejected/' + business.id), business);
    };

    /** Function to read loved businesses from Firebase */
    const readLovedBusinesses = async () => {
        const dbRef = ref(database);
        const snapshot = await get(child(dbRef, 'loved'));
        if (snapshot.exists()) {
            return snapshot.val();
        } else {
            console.log("No loved businesses available");
            return [];
        }
    };

    /** Function to read rejected businesses from Firebase */
    const readRejectedBusinesses = async () => {
        const dbRef = ref(database);
        const snapshot = await get(child(dbRef, 'rejected'));
        if (snapshot.exists()) {
            return snapshot.val();
        } else {
            console.log("No rejected businesses available");
            return [];
        }
    };

    /** The event that happens on a right swipe */
    const rightSwipe = () => {
        const lovedBusiness = props.businesses[currentB];
        if (loved.length === undefined) {
            setLoved([lovedBusiness]);
        } else {
            setLoved([...loved, lovedBusiness]);
        }
        saveLovedBusiness(lovedBusiness);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
        console.log(currentB);
        if (currentB === 5){
            console.log("hey wassup");
            fetchBusinesses();
        }
    };

    /** The event that happens on a left swipe */
    const leftSwipe = () => {
        const rejectedBusiness = props.businesses[currentB];
        if (rejected.length === undefined) {
            setRejected([rejectedBusiness]);
        } else {
            setRejected([...rejected, rejectedBusiness]);
        }
        saveRejectedBusiness(rejectedBusiness);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
        console.log(currentB);
        if (currentB === 5){
            console.log("hey wassup");
            fetchBusinesses();
        }
    };

    const handleTouchStart = (e: React.TouchEvent) => {
        touchStartX.current = e.targetTouches[0].clientX;
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        touchEndX.current = e.targetTouches[0].clientX;
    };

    const handleTouchEnd = () => {
        if (touchStartX.current - touchEndX.current > 50) {
            leftSwipe();
        }

        if (touchEndX.current - touchStartX.current > 50) {
            rightSwipe();
        }
    };

    useEffect(() => {
        const loadBusinesses = async () => {
            const lovedBusinesses: Business[] = await readLovedBusinesses();
            const rejectedBusinesses: Business[] = await readRejectedBusinesses();
            setLoved(lovedBusinesses);
            setRejected(rejectedBusinesses);
        };
        loadBusinesses();
    }, []);

    if (currentB === 5){
        return (
            <div>
                <p>Loading...</p>
                <button onClick={fetchBusinesses}>Fetch More</button>
            </div>

        );
       
    }

    /** After all businesses have been filtered through or there are none avalible  */
    if (currentB === -1 || props.businesses.length <= 0) {
        return (<div>
            <h1 className="text-2xl text-center">Congrats! You've swiped through all the businesses in your area!</h1>
        </div>);
    }

    return (
        <div
            className="relative min-h-screen w-full"
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
        >
            <BusinessCard {...props.businesses[currentB]} />
            <div className="flex justify-between fixed bottom-0 left-0 right-0 p-4 bg-[#8ca9ad] z-50">
              {isDesktop && (
                <div className="flex w-full">
                  <button onClick={leftSwipe} className="flex-1 bg-red-500 text-white py-2 px-4 rounded hover:bg-gray-700 outline outline-[#f4f4f4]">
                    <ClearIcon />
                  </button>
                  &nbsp;&nbsp;
                  <button onClick={rightSwipe} className="flex-1 bg-green-500 text-white py-2 px-4 rounded hover:bg-gray-700 outline outline-[#f4f4f4]">
                    <CheckIcon />
                  </button>
                </div>
              )}
            </div>
            <div style={{ paddingBottom: "42px" }}></div>
        </div>
    );
};